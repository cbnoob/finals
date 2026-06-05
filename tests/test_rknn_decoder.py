"""
Tests for the YOLOv11 RKNN decoder using synthetic inference tensors.
No NPU required — we hand-build the raw output the NPU would produce.
"""

import numpy as np

from detection.rknn_decoder import decode_yolov11_rknn, sigmoid, xywh2xyxy


def _make_output(num_classes=80, num_boxes=8400):
    """One strong detection, rest background. Shape (1, 4+num_classes, num_boxes)."""
    ch = 4 + num_classes
    pred = np.full((ch, num_boxes), 0.0, dtype=np.float32)
    # Background: very negative class logits -> sigmoid ~0
    pred[4:, :] = -10.0
    # Box 0: centered object at (320,320) size 100x100, class 2 strong
    pred[0, 0] = 320.0
    pred[1, 0] = 320.0
    pred[2, 0] = 100.0
    pred[3, 0] = 100.0
    pred[4 + 2, 0] = 10.0  # class 2 logit high
    return [pred[np.newaxis, ...]]  # (1, ch, num_boxes)


def test_sigmoid_range():
    out = sigmoid(np.array([-10.0, 0.0, 10.0]))
    assert out[0] < 0.01 and abs(out[1] - 0.5) < 1e-6 and out[2] > 0.99


def test_xywh2xyxy():
    boxes = np.array([[100.0, 100.0, 50.0, 20.0]])
    xyxy = xywh2xyxy(boxes)
    assert list(xyxy[0]) == [75.0, 90.0, 125.0, 110.0]


def test_single_detection_decoded():
    outputs = _make_output()
    boxes, scores, classes = decode_yolov11_rknn(
        outputs, img_shape=(640, 640, 3), model_input_size=(640, 640)
    )
    assert len(boxes) == 1
    assert int(classes[0]) == 2
    assert scores[0] > 0.9
    x1, y1, x2, y2 = boxes[0]
    assert abs(x1 - 270) < 2 and abs(y1 - 270) < 2
    assert abs(x2 - 370) < 2 and abs(y2 - 370) < 2


def test_scaling_to_original_image():
    outputs = _make_output()
    # Original image twice the model input -> coords should double
    boxes, _, _ = decode_yolov11_rknn(
        outputs, img_shape=(1280, 1280, 3), model_input_size=(640, 640)
    )
    x1, y1, x2, y2 = boxes[0]
    assert abs(x1 - 540) < 4 and abs(x2 - 740) < 4


def test_confidence_filter_removes_weak():
    ch = 84
    pred = np.full((ch, 8400), 0.0, dtype=np.float32)
    pred[4:, :] = -10.0  # all background, sigmoid ~ 0
    boxes, scores, classes = decode_yolov11_rknn(
        [pred[np.newaxis, ...]], img_shape=(640, 640, 3)
    )
    assert len(boxes) == 0
