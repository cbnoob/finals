"""
YOLOv11 RKNN output decoder.

Adapted from the organizer's rknndecoder.py (reference/organizer_samples/).
Pure numpy/OpenCV — no NPU needed, so it is unit-testable on any laptop.

YOLOv11 RKNN exports output raw logits for the class scores, so we apply
sigmoid here (this is the key difference from the YOLOv8 single-image sample).
"""

from __future__ import annotations

import cv2
import numpy as np

CONF_THRES = 0.25
IOU_THRES = 0.45


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=int)
    # NMSBoxes expects [x, y, w, h]; convert from xyxy for correct IoU.
    wh_boxes = boxes.copy()
    wh_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    wh_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    idxs = cv2.dnn.NMSBoxes(
        wh_boxes.tolist(), scores.tolist(),
        score_threshold=0.0, nms_threshold=iou_thres,
    )
    if len(idxs) == 0:
        return np.array([], dtype=int)
    return np.array(idxs).flatten()


def decode_yolov11_rknn(
    outputs,
    img_shape,
    model_input_size=(640, 640),
    conf_thres: float = CONF_THRES,
    iou_thres: float = IOU_THRES,
):
    """
    outputs: RKNN inference outputs (list with one tensor)
    img_shape: original image shape (H, W) or (H, W, C)
    Returns: (boxes_xyxy, scores, class_ids) as numpy arrays.
    """
    pred = outputs[0]

    # Accept (1, 84, 8400) or (1, 8400, 84)
    if pred.shape[1] == 84 or (pred.ndim == 3 and pred.shape[1] < pred.shape[2]):
        pred = pred[0].transpose(1, 0)
    else:
        pred = pred[0]

    boxes = pred[:, :4]
    class_scores = sigmoid(pred[:, 4:])

    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes = xywh2xyxy(boxes)

    gain_w = img_shape[1] / model_input_size[0]
    gain_h = img_shape[0] / model_input_size[1]
    boxes[:, [0, 2]] *= gain_w
    boxes[:, [1, 3]] *= gain_h

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])

    idxs = nms_boxes(boxes, scores, iou_thres)
    if len(idxs) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    return boxes[idxs], scores[idxs], class_ids[idxs]


def draw_detections(img, boxes, scores, class_ids, class_names):
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        cls = int(cls)
        name = class_names[cls] if 0 <= cls < len(class_names) else f"id_{cls}"
        label = f"{name} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )
    return img
