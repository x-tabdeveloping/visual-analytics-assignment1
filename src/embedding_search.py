from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.pipeline import make_pipeline

from utils.embeddings import ImagePreprocessor, VGG16Encoder


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="embedding_search",
        description="Image retrieval with pretrained embeddings.",
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
    parser.add_argument(
        "-b",
        "--batch_size",
        default=16,
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    encoder = make_pipeline(
        ImagePreprocessor(desired_size=224), VGG16Encoder()
    )

    print("Finding image files.")
    image_files = Path(args.images_path).glob("*.jpg")
    image_files = list(image_files)

    print("Loading images")
    images = [np.array(Image.open(image_file)) for image_file in image_files]

    print("Encoding all other images.")
    image_embeddings = encoder.fit_transform(images)

    print("Encoding query image.")
    # Encoding the target image
    target_path = Path(args.query_image_path)
    target_image = np.array(Image.open(target_path))
    # We need to call fit() as it initializes the VGG16 model to the right image size
    target_embedding = encoder.transform([target_image])[0]

    print("Calculating distances to query image.")
    distances = [
        cosine(target_embedding, embedding) for embedding in image_embeddings
    ]

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
