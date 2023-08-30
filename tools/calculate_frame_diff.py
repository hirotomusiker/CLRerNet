import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="CULane frame value difference calculation"
    )
    parser.add_argument(
        "root_path",
        type=str,
        help="CULane dataset root path that contains 'list' folder.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    key = "train_gt"
    list_dir = Path(args.root_path).joinpath("list")
    list_path = str(list_dir.joinpath(f"{key}.txt"))
    out_path = str(list_dir.joinpath("train_diffs"))

    with open(list_path, "rb") as f:
        lines = f.readlines()

    prev_img = np.zeros(1)
    diffs = []
    for line in tqdm(lines):
        img_path = line.decode("utf8").strip().split(" ")[0][1:]
        path = str(Path(args.root_path).joinpath(img_path))
        img = cv2.imread(path)
        diffs.append(
            np.abs(img.astype(np.float32) - prev_img.astype(np.float32)).sum()
            / (img.shape[0] * img.shape[1] * img.shape[2])
        )
        prev_img = img

    # save the results as an npz file
    np.savez(out_path, data=diffs)


if __name__ == "__main__":
    main()
