"""Licence plate detection, reading (OCR) and anonymisation service."""
from io import BytesIO
from logging import info
from uuid import uuid4

import numpy as np
import onnxruntime as rt
from easyocr import Reader
from PIL import Image

from models.license_plate import LicensePlate, db
from services.utils import blur_part, non_max_suppression, process_predictions, scale

PLATE_MODEL_PATH = "weights/model.onnx"
TUNIS_MODEL_PATH = "weights/tunis.onnx"
PLATE_INPUT_SIZE = 256
TUNIS_INPUT_SIZE = 640
UPLOAD_DIR = "images"


class LicensePlateService:
    """Singleton wrapping the ONNX models and the OCR reader (loaded once)."""

    _instance = None

    def __init__(self):
        self.session = rt.InferenceSession(PLATE_MODEL_PATH)
        self.tunis_detector = rt.InferenceSession(TUNIS_MODEL_PATH)
        self.reader = Reader(["fr"])

    @staticmethod
    def get_instance():
        if LicensePlateService._instance is None:
            LicensePlateService._instance = LicensePlateService()
        return LicensePlateService._instance

    def detect_license_plate(self, pic):
        """Return the ``[x, y, w, h]`` box of the single plate in a PIL image."""
        ow, oh = pic.size
        image = pic.resize((PLATE_INPUT_SIZE, PLATE_INPUT_SIZE))
        image = np.array(image).astype(np.float32) / 255
        image = np.expand_dims(image, axis=0)

        output = self.session.run(None, {"input_1": image})[0]
        bboxes = np.round(process_predictions(output)[0]).astype(int).tolist()
        bboxes = [scale(ow, oh, PLATE_INPUT_SIZE, PLATE_INPUT_SIZE, b) for b in bboxes]

        if len(bboxes) != 1:
            raise Exception("error detecting license plate")
        info(f"detecting license plate {bboxes[0]}")
        return bboxes[0]

    def detect_license_plates(self, images):
        """Return, for each uploaded file, the list of detected ``[x, y, w, h]`` boxes."""
        pics = []
        sizes = []
        for file in images:
            image = Image.open(BytesIO(file.read()))
            sizes.append(image.size)
            image = image.resize((PLATE_INPUT_SIZE, PLATE_INPUT_SIZE))
            pics.append(np.array(image).astype(np.float32) / 256)

        output = self.session.run(None, {"input_1": np.array(pics)})[0]
        bboxess = np.round(process_predictions(output)).astype(int).tolist()
        for i, (ow, oh) in enumerate(sizes):
            bboxess[i] = [
                scale(ow, oh, PLATE_INPUT_SIZE, PLATE_INPUT_SIZE, b) for b in bboxess[i]
            ]
        return bboxess

    def read_license_plate(self, image):
        """Read the plate number from a cropped plate image."""
        bboxes = self.detect_tunis(image)
        left, right = self.split_plate(image, bboxes)
        return f"{self.read_image(left)} TN {self.read_image(right)}"

    def detect_tunis(self, pic):
        """Locate the "تونس" band in a cropped plate; return ``[x, y, w, h]``."""
        ow, oh = pic.size
        image = pic.resize((TUNIS_INPUT_SIZE, TUNIS_INPUT_SIZE))
        image = np.array(image).astype(np.float32) / 255
        image = np.array([np.transpose(image, (2, 0, 1))])

        out = self.tunis_detector.run(None, {"images": image})[0][0]
        indices = non_max_suppression(out[:, 0:4], out[:, 4], 0.5, 0.5)
        boxes = out[indices]
        info(f"tunis model output {boxes}")
        if len(boxes) != 1:
            info(f"detecting tunis output length is {len(boxes)}")
            raise Exception("error reading license plate")

        x, y, w, h, _, _ = boxes[0]
        # scale to original image, then convert centre format to top-left
        x = (x * ow) / TUNIS_INPUT_SIZE
        y = (y * oh) / TUNIS_INPUT_SIZE
        w = (w * ow) / TUNIS_INPUT_SIZE
        h = (h * oh) / TUNIS_INPUT_SIZE
        box = [int(x - w / 2), int(y - h / 2), int(w), int(h)]
        info(f"detecting tunis {box}")
        return box

    def split_plate(self, pic, bbox):
        """Split a plate image into the parts left and right of the "تونس" band."""
        x, _, w, _ = bbox
        array = np.array(pic)
        left = Image.fromarray(array[:, :x, :])
        right = Image.fromarray(array[:, x + w:, :])
        info("splitting image")
        return left, right

    def read_image(self, image):
        """OCR the first text found in a PIL image."""
        text = self.reader.readtext(np.array(image), detail=0)[0]
        info(f"reading text {text}")
        return text

    def select_image(self, bboxes):
        """Order image indices by plate width/height ratio, widest first."""
        indices = sorted(
            range(len(bboxes)), key=lambda i: bboxes[i][2] / bboxes[i][3], reverse=True
        )
        info(f"selecting image index {indices}")
        return indices

    def process_images(self, images_fs):
        """Full pipeline: detect, read, blur, save and persist. Returns ``{id, license_plate}``."""
        images = [Image.open(BytesIO(fs.read())) for fs in images_fs]
        bboxes = [self.detect_license_plate(image) for image in images]

        # read from the first image (widest plate first) where "تونس" is found
        selected, tbbox, plate = None, None, None
        for idx in self.select_image(bboxes):
            x, y, w, h = bboxes[idx]
            crop = images[idx].crop((x, y, x + w, y + h))
            try:
                tbbox = self.detect_tunis(crop)
            except Exception:
                continue
            selected, plate = idx, crop
            break
        if selected is None:
            raise Exception("error reading license plate number [detecting Tunis]")
        info(f"selected image index {selected}")

        left, right = self.split_plate(plate, tbbox)
        license_plate_text = f"{self.read_image(left)} TN {self.read_image(right)}"

        # anonymise and store the images
        paths = ""
        for image, bbox in zip(images, bboxes):
            filename = str(uuid4())
            blur_part(image, bbox).save(f"{UPLOAD_DIR}/{filename}.jpg")
            paths += f"{filename}$"

        record = LicensePlate(license_plate=license_plate_text, paths=paths)
        db.session.add(record)
        db.session.commit()
        return {"id": record.id, "license_plate": license_plate_text}
