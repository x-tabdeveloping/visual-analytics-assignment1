from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Index of the chosen image
CHOSEN_IMAGE = 0


def extract_histogram_features(image: np.ndarray) -> np.ndarray:
    """This creates a histogram for each channel
    and concatenates them into one vector"""
    channel_histograms = []
    bins = np.arange(start=0, stop=256, step=1)
    for channel in image.transpose((2, 0, 1)):
        hist, _ = np.histogram(channel, bins=bins, range=(0, 255))
        channel_histograms.append(hist)
    return np.concatenate(channel_histograms)


def chisqr_distance(h1: np.ndarray, h2: np.ndarray):
    """I copied the implementation from OpenCV,
    because I don't wanna use OpenCV."""
    # Avoiding division with zero
    mask = h1 != 0
    h1 = h1[mask]
    h2 = h2[mask]
    return np.sum(np.square(h1 - h2) / h1)


print("Finding image files.")
image_files = Path("data/jpg").glob("image_*.jpg")
image_files = list(sorted(image_files))

print("Loading images")
images = [np.array(Image.open(file)) for file in image_files]
histograms = [extract_histogram_features(image) for image in images]

print("Calculating distances.")
chosen = histograms[CHOSEN_IMAGE]
distances = np.array([chisqr_distance(chosen, other) for other in tqdm(histograms)])
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
    out_df["Filename"] == image_files[CHOSEN_IMAGE], "Target"
)

out_folder = Path("out")
out_folder.mkdir(exist_ok=True)
target_id = image_files[CHOSEN_IMAGE].stem
out_file = out_folder.joinpath(f"{target_id}.csv")
out_df.to_csv(out_file)
