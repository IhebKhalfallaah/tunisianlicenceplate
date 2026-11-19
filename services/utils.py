"""Pure image / bounding-box helpers used by the licence plate service."""
import numpy as np
from PIL import ImageFilter

GRID_SIZE = 32
MODEL_INPUT_SIZE = 256


def scale(ow, oh, nw, nh, bbox):
    """Scale an ``[x, y, w, h]`` box from an ``nw x nh`` image to an ``ow x oh`` image."""
    x, y, w, h = bbox
    return [(x * ow) // nw, (y * oh) // nh, (w * ow) // nw, (h * oh) // nh]


def blur_part(image, bbox):
    """Blur the ``[x, y, w, h]`` region of a PIL image in place and return the image."""
    left, top, width, height = bbox
    region = image.crop([left, top, left + width, top + height])
    for _ in range(5):
        region = region.filter(ImageFilter.GaussianBlur(20)).convert("RGBA")
    image.paste(region, (left, top))
    return image


def compute_overlap(box1, box2):
    """Intersection over union of two ``[x, y, h, w]`` boxes."""
    x, y, h, w = box1
    box1 = [x, y, w + x, h + y]
    x, y, h, w = box2
    box2 = [x, y, w + x, h + y]

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    return intersection / union


def non_max_suppression(boxes, scores, conf_threshold, overlap_threshold):
    """Return indices of the boxes kept by non-maximum suppression, best score first."""
    candidates = [i for i in range(len(boxes)) if scores[i] >= conf_threshold]
    remaining = sorted(candidates, key=lambda i: scores[i], reverse=True)

    selected = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [
            i for i in remaining
            if compute_overlap(boxes[current], boxes[i]) <= overlap_threshold
        ]
    return selected


def create_image_grid():
    """Grid of ``[x, y, cell_w, cell_h]`` cells covering the 256x256 model input."""
    cell_size = MODEL_INPUT_SIZE / GRID_SIZE
    image_grid = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            image_grid[i, j] = [j * cell_size, i * cell_size, cell_size, cell_size]
    return image_grid


def prediction_to_bbox(bboxes, image_grid):
    """Convert grid-relative predictions to absolute ``[x_min, y_min, x_max, y_max]``."""
    bboxes = bboxes.copy()
    im_width = image_grid[:, :, 2] * GRID_SIZE
    im_height = image_grid[:, :, 3] * GRID_SIZE

    # descale x, y
    bboxes[:, :, 1] = (bboxes[:, :, 1] * image_grid[:, :, 2]) + image_grid[:, :, 0]
    bboxes[:, :, 2] = (bboxes[:, :, 2] * image_grid[:, :, 3]) + image_grid[:, :, 1]
    # descale width, height
    bboxes[:, :, 3] = bboxes[:, :, 3] * im_width
    bboxes[:, :, 4] = bboxes[:, :, 4] * im_height
    # centre x, y -> top-left x, y
    bboxes[:, :, 1] = bboxes[:, :, 1] - (bboxes[:, :, 3] / 2)
    bboxes[:, :, 2] = bboxes[:, :, 2] - (bboxes[:, :, 4] / 2)
    # width, height -> x_max, y_max
    bboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3]
    bboxes[:, :, 4] = bboxes[:, :, 2] + bboxes[:, :, 4]
    return bboxes


def switch_x_y(bboxes):
    """Swap x and y coordinates of ``[x1, y1, x2, y2]`` boxes (returns a copy)."""
    return bboxes[:, [1, 0, 3, 2]]


def _grid_nms(predictions):
    """Apply NMS to the grid predictions of a single image."""
    probabilities = predictions[:, :, 0].flatten()
    bboxes = predictions[:, :, 1:5].reshape((GRID_SIZE * GRID_SIZE, 4))
    bboxes = switch_x_y(bboxes)
    top_indices = non_max_suppression(
        boxes=bboxes, scores=probabilities, conf_threshold=0.3, overlap_threshold=0.3
    )
    return switch_x_y(bboxes[top_indices])


def process_predictions(predictions):
    """Turn raw detector output into a list (per image) of ``[x, y, w, h]`` boxes."""
    image_grid = create_image_grid()
    results = []
    for prediction in predictions:
        bboxes = _grid_nms(prediction_to_bbox(prediction, image_grid))
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        results.append(bboxes)
    return results
