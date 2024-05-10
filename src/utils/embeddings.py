import numpy as np
from PIL import Image
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.vgg16 import VGG16


def pad_to_square(im, size: int):
    """Pads an image to a given sized square."""
    thumbnail = im.copy()
    thumbnail.thumbnail((size, size))
    width, height = thumbnail.size
    if width == height:
        return thumbnail
    elif width > height:
        result = Image.new(thumbnail.mode, (width, width))
        result.paste(thumbnail, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(thumbnail.mode, (height, height))
        result.paste(thumbnail, ((height - width) // 2, 0))
        return result


class ImagePreprocessor(TransformerMixin):
    """Pads image to desired sized squares
    and divides pixel values by 255."""

    def __init__(self, desired_size: int):
        self.desired_size = desired_size

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = []
        for im in X:
            im = Image.fromarray(im)
            im = pad_to_square(im, self.desired_size)
            im = np.array(im)
            result.append(im)
        result = np.stack(result)
        result = (result / 255).astype(np.float32)
        return result


class VGG16Encoder(TransformerMixin):
    """Encodes images with VGG16."""

    def __init__(
        self,
    ):
        self.model = None

    def fit(self, X: np.ndarray, y=None):
        self.input_shape = tuple(X.shape[1:])
        self.model = VGG16(
            input_shape=self.input_shape,
            weights="imagenet",
            pooling="avg",
            include_top=False,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        embeddings = np.array(self.model.predict(X))
        return embeddings
