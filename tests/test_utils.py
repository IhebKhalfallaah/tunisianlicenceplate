import numpy as np
from PIL import Image

from services.utils import (
    blur_part,
    compute_overlap,
    create_image_grid,
    non_max_suppression,
    scale,
)


def test_scale_maps_bbox_from_model_size_to_original_size():
    assert scale(512, 256, 256, 256, [10, 20, 30, 40]) == [20, 20, 60, 40]


def test_overlap_of_identical_boxes_is_one():
    assert compute_overlap([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_overlap_of_disjoint_boxes_is_zero():
    assert compute_overlap([0, 0, 10, 10], [50, 50, 10, 10]) == 0.0


def test_nms_keeps_best_box_and_drops_overlapping_and_low_scores():
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [100, 100, 10, 10], [200, 200, 10, 10]]
    scores = [0.6, 0.9, 0.7, 0.1]
    assert non_max_suppression(boxes, scores, 0.5, 0.5) == [1, 2]


def test_image_grid_has_32x32_cells_of_8_pixels():
    grid = create_image_grid()
    assert grid.shape == (32, 32, 4)
    assert grid[0, 1, 0] == 8
    assert grid[1, 0, 1] == 8


def test_blur_part_keeps_image_size():
    image = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
    result = blur_part(image, [10, 20, 50, 30])
    assert result.size == (200, 100)
