from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from utils.embeddings import ImagePreprocessor, VGG16Encoder
from utils.hist import HistogramEncoder, chisqr_distance

embedding_pipelines = {
    "vgg16": make_pipeline(
        ImagePreprocessor(desired_size=224), VGG16Encoder()
    ),
    "hist": HistogramEncoder(),
}

distance_metrics = {"chisqr": chisqr_distance, "cosine": cosine}


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="image_search",
        description="Image retrieval with pretrained embeddings or histogram comparison.",
    )
    parser.add_argument("image_id")
    parser.add_argument(
        "-s",
        "--source_path",
        default="data/jpg",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        default="out",
    )
    parser.add_argument(
        "-r",
        "--representation",
        default="hist",
    )
    parser.add_argument("-d", "--distance_metric", default="chisqr")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    try:
        encoder = embedding_pipelines[args.representation]
    except KeyError as e:
        valid_options = " ".join(embedding_pipelines.keys())
        raise KeyError(
            f"Representation should be one of {{{valid_options}}}, given: {args.representation}"
        ) from e
    try:
        distance_metric = distance_metrics[args.distance_metric]
    except KeyError as e:
        valid_options = " ".join(distance_metrics.keys())
        raise KeyError(
            f"distance_metric should be one of {{{valid_options}}}, given: {args.distance_metric}"
        ) from e

    print("Finding image files.")
    image_files = Path(args.source_path).glob("*.jpg")
    image_files = list(sorted(image_files))
    image_ids = [filename.stem for filename in image_files]
    id_to_idx = dict(zip(image_ids, range(len(image_ids))))
    try:
        selected_image_idx = id_to_idx[args.image_id]
    except KeyError as e:
        raise KeyError(
            f"Specified image with ID {args.image_id} not found."
        ) from e

    print("Loading images")
    images = [np.array(Image.open(file)) for file in tqdm(image_files)]

    print("Representing images")
    embeddings = encoder.fit_transform(images)

    print("Calculating distances.")
    chosen = embeddings[selected_image_idx]
    distances = np.array(
        [distance_metric(chosen, other) for other in tqdm(embeddings)]
    )
    top_indices = np.argsort(distances)[: 5 + 1]

    print("Saving results.")
    out_df = pd.DataFrame(
        {
            "Filename": [image_files[i] for i in top_indices],
            "Distance": [distances[i] for i in top_indices],
        }
    )
    # Renaming the chosen image to target
    out_df["Filename"] = out_df["Filename"].mask(
        out_df["Filename"] == image_files[selected_image_idx], "Target"
    )

    out_folder = Path(args.out_path)
    out_folder.mkdir(exist_ok=True)
    target_id = image_files[selected_image_idx].stem
    out_file = out_folder.joinpath(
        f"{target_id}_{args.representation}_{args.distance_metric}.csv"
    )
    out_df.to_csv(out_file)


if __name__ == "__main__":
    main()
