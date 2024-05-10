import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import minmax_scale


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


class HistogramEncoder(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        hists = np.stack([extract_histogram_features(image) for image in X])
        return minmax_scale(hists, axis=1)
