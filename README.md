[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrernet-improving-confidence-of-lane/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=clrernet-improving-confidence-of-lane)

# CLRerNet Official Implementation

The official implementation of [our paper](https://openaccess.thecvf.com/content/WACV2024/html/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.html) "CLRerNet: Improving Confidence of Lane Detection with LaneIoU" (WACV 2024), by Hiroto Honda and Yusuke Uchida.

## What's New

- **(dev) LaneATT with LaneIoU-based matcher [configs](configs/laneatt/README.md)**
- **Draft branches under development: [[CurveLanes dataset](https://github.com/hirotomusiker/CLRerNet/tree/feature/curvelanes)]**
- **[v0.3.0 release](https://github.com/hirotomusiker/CLRerNet/tree/v0.3.0) supports mmdet3x environment!**
- Code for training is available ! (Dec. 1, 2023)
- Our CLRerNet paper has been accepted to WACV2024 ! (Oct. 25, 2023)
- LaneIoU loss and cost are published. ([PR#17](https://github.com/hirotomusiker/CLRerNet/pull/17), Oct.22, 2023)


## Method

CLRerNet features LaneIoU for the target assignment cost and loss functions aiming at the improved quality of confidence scores.<br>
LaneIoU takes the local lane angles into consideration to better correlate with the segmentation-based IoU metric.

<p align="left"> <img src="docs/figures/clrernet.jpg" height="200"\></p>
<p align="left"> <img src="docs/figures/laneiou.jpg" height="160"\></p>

## Performance

CLRerNet achieves the <b>state-of-the-art performance on CULane benchmark </b> significantly surpassing the baseline.

Model           | Backbone | F1 score | GFLOPs
---             | ---      | ---           | ---
CLRNet        | DLA34    | 80.47  | 18.4
[CLRerNet](https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34.pth)        | DLA34    | 81.12&pm;0.04 <sup>*</sup>| 18.4
[CLRerNet&#8902;](https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34_ema.pth) | DLA34    | 81.43&pm;0.14 <sup>*</sup> | 18.4


\* F1 score stats of five models reported in our paper. The release models' scores are 81.11 (CLRerNet) and 81.55 (CLRerNet&#8902;, EMA model) respectively.

## Install

This repo is now based on the [mmdetection 3.3](https://github.com/open-mmlab/mmdetection/tree/v3.3.0) environment.
If you prefer the previous mmdet2x-based CLRerNet, please checkout the [v0.2.1 branch](https://github.com/hirotomusiker/CLRerNet/tree/v0.2.1).

Docker environment is recommended for installation:
```bash
docker compose build --build-arg UID="`id -u`" dev
docker compose run --rm dev
```

See [Installation Tips](docs/INSTALL.md) for more details.

## Inference

Run the following command to detect the lanes from the image and visualize them:
```bash
python demo/image_demo.py demo/demo.jpg configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file=result.png
```

## Test

Run the following command to evaluate the model on CULane dataset:

```bash
python tools/test.py configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth
```

For dataset preparation, please refer to [Dataset Preparation](docs/DATASETS.md).

## Frame Difference Calculation

Filtering out redundant frames during training helps the model avoid overfitting to them. We provide a simple calculator that outputs an npz file containing frame difference values.

```bash
python tools/calculate_frame_diff.py [culane_root_path]
```

Also you can find the npz file [[here]](https://github.com/hirotomusiker/CLRerNet/releases/download/v0.2.0/train_diffs.npz).


## Train

Make sure that the frame difference npz file is prepared as `dataset/culane/list/train_diffs.npz`.<br>
Run the following command to train a model on CULane dataset:

```bash
python tools/train.py configs/clrernet/culane/clrernet_culane_dla34.py
```

## Speed Test

Calculate fps by inference iteration.

```bash
python tools/speed_test.py configs/clrernet/culane/clrernet_culane_dla34.py clrernet_culane_dla34.pth --filename demo/demo.jpg --n_iter_warmup 1000 --n_iter_test 10000
```

## Citation

```BibTeX
@inproceedings{honda2024clrernet,
  title={Clrernet: improving confidence of lane detection with laneiou},
  author={Honda, Hiroto and Uchida, Yusuke},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={1176--1185},
  year={2024}
}
```

## References

* [Turoad/CLRNet](https://github.com/Turoad/CLRNet/)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
* [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [optn-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
