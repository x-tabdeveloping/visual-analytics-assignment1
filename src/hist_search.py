from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def image_histogram(image: np.ndarray) -> np.ndarray:
    """Calculates normalized histogram for given image."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [256] * 3, [0, 256] * 3)
    hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return hist


def chisqr_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Calculates chi square histogram distance."""
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="hist_search",
        description="Image retrieval with histogram comparison.",
    )
    parser.add_argument("query_image_path")
    parser.add_argument(
        "-i",
        "--images_path",
        default="data/jpg",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        default="out",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        default=5,
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    print("Finding image files.")
    image_files = Path(args.images_path).glob("*.jpg")
    image_files = list(image_files)

    print("Encoding query image.")
    target_path = Path(args.query_image_path)
    target_image = cv2.imread(str(target_path))
    target_hist = image_histogram(target_image)
    distances = []
    for image_file in tqdm(
        image_files, desc="Calculating distances to query image."
    ):
        image = cv2.imread(str(image_file))
        hist = image_histogram(image)
        distances.append(chisqr_distance(target_hist, hist))

    print("Selecting closest images.")
    top_indices = np.argsort(distances)[: args.top_k + 1]

    print("Saving results.")
    out_df = pd.DataFrame(
        {
            "Filename": [image_files[i].name for i in top_indices],
            "Distance": [distances[i] for i in top_indices],
        }
    )

    out_folder = Path(args.out_path)
    out_folder.mkdir(exist_ok=True, parents=True)
    out_file = out_folder.joinpath(f"{target_path.stem}.csv")
    out_df.to_csv(out_file)


if __name__ == "__main__":
    main()
