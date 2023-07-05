"""
Adapted from:
https://github.com/lucastabelini/LaneATT/blob/main/utils/culane_metric.py
Copyright (c) 2021 Lucas Tabelini
"""

import os
from pathlib import Path
from functools import partial

import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.optimize import linear_sum_assignment

from mmcv.utils import print_log

from libs.utils.visualizer import draw_lane
from libs.utils.lane_utils import interp


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    """
    Args:
        xs (np.ndarray): Array containing lane coordinate arrays with different lengths.
        ys (np.ndarray): Array containing lane coordinate arrays with different lengths.
        width (int): Lane drawing width to calculate IoU.
        img_shape (tuple): Image shape.
    Returns:
        ious (np.ndarray): IoU matrix.

    """
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]
    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def culane_metric(
    pred,
    anno,
    cat,
    width=30,
    iou_thresholds=[0.5],
    img_shape=(590, 1640, 3),
):
    """
    Calculate the CULane metric for given pred and anno lanes of one image.
    Example of an IoU matrix and assignment:
     ious = [[0.85317694 0.         0.        ]
             [0.         0.49573853 0.        ]]
     (row_ind, col_ind) = (0, 0), (1, 1)
    Args:
        pred (List[List[tuple]]): Prediction result for one image.
        anno (List[List[tuple]]): Lane labels for one image.
        cat (str): Category name.
        width (int): Lane drawing width to calculate IoU.
        iou_thresholds (list): IoU thresholds for evaluation.
        img_shape (tuple): Image shape.
    Returns:
        results (dict) containing:
            n_gt (int): number of annotations
            cat (str): category name
            hits (List[np.ndarray]): bool array (TP or not)
    """
    interp_pred = np.array(
        [interp(pred_lane, n=5) for pred_lane in pred], dtype=object
    )
    interp_anno = np.array(
        [interp(anno_lane, n=5) for anno_lane in anno], dtype=object
    )

    ious = discrete_cross_iou(
        interp_pred, interp_anno, width=width, img_shape=img_shape
    )

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    pred_ious = np.zeros(len(pred))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    hits = [pred_ious > thr for thr in iou_thresholds]

    results = {
        'n_gt': len(anno),
        'cat': cat,
        'hits': hits,
    }

    return results


def load_culane_img_data(path):
    """
    Args:
        path (str): Txt file path containing lane coordinates.
    Returns:
        img_data (List[List[tuple]]): List of lane coordinates.
            Example: [
                         [(x_00, y_00), (x_01, y_01),....],
                         [(x_10, y_10), (x_11, y_11),....],
                     ]
    """
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]

    img_data = [
        [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data
    ]
    img_data = [lane for lane in img_data if len(lane) >= 2]
    return img_data


def load_culane_data(data_dir, file_list_path, data_cats):
    """
    Make the annotation list of all the test data
    and assign category names.
    Args:
        data_dir (str): Directory where the (prediction | test) txt files are stored.
        file_list_path (str): Test set data list path.
        data_cats (list): Category names.
    Returns:
        data (List[List[List[tuple]]]): Lists of list(s) of lane coordinates for all the images.
        cats (List[str]): List of category names.
    """
    with open(file_list_path, 'r') as file_list:
        data_paths = [
            line[1 if line[0] == '/' else 0 :].rstrip()
            for line in file_list.readlines()
        ]
        cats = [
            data_cats[p] if p in data_cats.keys() else 'test0_normal'
            for p in data_paths
        ]
        file_paths = [
            os.path.join(data_dir, line.replace('.jpg', '.lines.txt'))
            for line in data_paths
        ]

    data = []
    for path in tqdm(file_paths):
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data, cats


def load_categories(categories_path):
    """
    Load the test file list for each scene category.
    Args:
        categories_path (str): Dataset's category list path.
    Returns:
        data_cats (dict): Correspondence between image data path and category.
        categories (list): Category name list.
    """
    data_cats = {}
    categories = [
        'test0_normal',
        'test1_crowd',
        'test2_hlight',
        'test3_shadow',
        'test4_noline',
        'test5_arrow',
        'test6_curve',
        'test7_cross',
        'test8_night',
    ]
    for category in categories:
        with open(
            Path(categories_path).joinpath(category).with_suffix('.txt'), 'r'
        ) as file_list:
            data_cats.update({k.rstrip(): category for k in file_list.readlines()})
    return data_cats, categories


def eval_predictions(
    pred_dir,
    anno_dir,
    list_path,
    categories_dir,
    iou_thresholds=[0.1, 0.5, 0.75],
    width=30,
    sequential=False,
    logger=None,
):
    """
    Evaluate predictions on CULane dataset.
    Args:
        pred_dir (str): Directory where the prediction txt files are stored.
        anno_dir (str): Directory where the test labels are stored.
        list_path (str): Test set data list path.
        categories_dir (str): Dataset's category list directory.
        iou_thresholds (list): IoU threshold list for TP counting.
        width (int): Lane drawing width to calculate IoU.
        sequential (bool): Evaluate image-level results sequentially.
        logger (logging.Logger): Print to the mmcv log if not None.
    Returns:
        result_dict (dict): Evaluation result dict containing
            F1, precision, recall, etc. on the specified IoU thresholds.
    """
    print_log('List: {}'.format(list_path), logger=logger)
    data_cats, categories = load_categories(categories_dir)
    print_log('Loading prediction data...', logger=logger)
    predictions, _ = load_culane_data(pred_dir, list_path, data_cats)  # (34680,)
    print_log('Loading annotation data...', logger=logger)
    annotations, cats = load_culane_data(anno_dir, list_path, data_cats)  # (34680,)
    print_log(
        'Calculating metric {}...'.format(
            'sequentially' if sequential else 'in parallel'
        ),
        logger=logger,
    )
    img_shape = (590, 1640, 3)
    eps = 1e-8
    if sequential:
        results = t_map(
            partial(
                culane_metric,
                width=width,
                iou_thresholds=iou_thresholds,
                img_shape=img_shape,
            ),
            predictions,
            annotations,
            cats,
        )
    else:
        results = p_map(
            partial(
                culane_metric,
                width=width,
                iou_thresholds=iou_thresholds,
                img_shape=img_shape,
            ),
            predictions,
            annotations,
            cats,
        )

    result_dict = {}

    for k, iou_thr in enumerate(iou_thresholds):
        print_log(f"Evaluation results for IoU threshold = {iou_thr}", logger=logger)
        for i in range(len(categories) + 1):
            category = categories if i == 0 else [categories[i - 1]]
            n_gt_list = [r['n_gt'] for r in results if r['cat'] in category]
            n_category = len([r for r in results if r['cat'] in category])
            if n_category == 0:
                continue
            n_gts = sum(n_gt_list)
            hits = np.concatenate(
                [r['hits'][k] for r in results if r['cat'] in category]
            )
            tp = np.sum(hits)
            fp = len(hits) - np.sum(hits)
            prec = tp / (tp + fp + eps)
            rec = tp / (n_gts + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)

            if i == 0:
                cat_name = "test_all"
                result_dict.update(
                    {
                        f"TP{iou_thr}": tp,
                        f"FP{iou_thr}": fp,
                        f"FN{iou_thr}": n_gts - tp,
                        f"Precision{iou_thr}": prec,
                        f"Recall{iou_thr}": rec,
                        f"F1_{iou_thr}": f1,
                    }
                )
            else:
                cat_name = category[0]
                result_dict.update({f"F1_{cat_name}_{iou_thr}": f1})

            print_log(
                f"Eval category: {cat_name:12}, N: {n_category:4}, TP: {tp:5}, "
                f"FP: {fp:5}, FN: {n_gts-tp:5}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
                f"F1: {f1:.4f}",
                logger=logger,
            )

    return result_dict
