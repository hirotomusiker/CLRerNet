
# Dataset Preparation

## CULane

### Prepare CULane dataset
Firstly make the dataset directory at your home directory:
```
$HOME/dataset/culane
```

[\[Website\]](https://xingangpan.github.io/projects/CULane.html)
[\[Download page\]](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)

Download the tar.gz files from the [official gdrive](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)
, or use gdown as follows (cited from [LaneATT's docs](https://github.com/lucastabelini/LaneATT/blob/main/DATASETS.md#culane)):

```
gdown "https://drive.google.com/uc?id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8"
gdown "https://drive.google.com/uc?id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV"
gdown "https://drive.google.com/uc?id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7"
gdown "https://drive.google.com/uc?id=1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu"
gdown "https://drive.google.com/uc?id=18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd"
```
Then extract the downloaded files in the dataset directory:
```
tar xf driver_37_30frame.tar.gz
tar xf driver_100_30frame.tar.gz
tar xf driver_193_90frame.tar.gzt
tar xf annotations_new.tar.gz
tar xf list.tar.gz
```
Finally the dataset folder would look like:
```
$HOME/dataset/culane/
├── driver_100_30frame
├── driver_193_90frame
├── driver_37_30frame
└── list
```


