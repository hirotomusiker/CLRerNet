import copy
import numpy as np
import cv2

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)


def draw_lane(lane, img=None, img_shape=None, width=30, color=(255, 255, 255)):
    """
    Overlay lanes on the image.
    Args:
        lane (np.ndarray): N (x, y) coordinates from a single lane, shape (N, 2).
        img (np.ndarray): Source image.
        img_shape (tuple): Blank image shape used when img is None.
        width (int): Lane thickness.
        color (tuple): Lane color in BGR.
    Returns:
        img (np.ndarray): Output image.
    """
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color, thickness=width)
    return img


def visualize_lanes(
    src,
    preds,
    annos=list(),
    pred_ious=None,
    iou_thr=0.5,
    concat_src=False,
    save_path=None,
):
    """
    visualize lane markers from prediction results and ground-truth labels
    Args:
        src (np.ndarray): Source image.
        preds (List[np.ndarray]): Lane predictions.
        annos (List[np.ndarray]): Lane annotations.
        pred_ious (List[np.ndarray]): Pred-GT IoUs.
        iou_thr (float): Positive threshold for pred-GT IoU.
        concat_src (bool): Concatenate the original and overlaid images vertically.
        save_path (str): The output image file path.
    Returns:
        dst (np.ndarray): Output image.
    """
    dst = copy.deepcopy(src)
    for anno in annos:
        dst = draw_lane(anno, dst, dst.shape, width=4, color=GT_COLOR)
    if pred_ious == None:
        hits = [True for i in range(len(preds))]
    else:
        hits = [iou > iou_thr for iou in pred_ious]
    for pred, hit in zip(preds, hits):
        color = PRED_HIT_COLOR if hit else PRED_MISS_COLOR
        dst = draw_lane(pred, dst, dst.shape, width=4, color=color)
    if concat_src:
        dst = np.concatenate((src, dst), axis=0)
    if save_path:
        cv2.imwrite(save_path, dst)
    return dst
