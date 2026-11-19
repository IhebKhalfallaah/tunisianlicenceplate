from io import BytesIO

from flask import render_template, request
from PIL import Image

from services.license_plate_service import LicensePlateService

service = LicensePlateService.get_instance()


def _collect_images():
    """Return uploaded files sent as ``image0``, ``image1``, ... (stops at the first gap)."""
    images = []
    while f"image{len(images)}" in request.files:
        images.append(request.files[f"image{len(images)}"])
    return images


def index():
    return render_template("index.html")


def about():
    return render_template("about.html")


def demo():
    return render_template("demo.html")


def detect_multiple():
    return {"bboxes": service.detect_license_plates(_collect_images())}


def detect():
    image = Image.open(BytesIO(request.files["image"].read()))
    return {"bboxes": service.detect_license_plate(image)}


def read():
    try:
        return service.process_images(_collect_images())
    except Exception as e:
        return {"error": str(e)}
