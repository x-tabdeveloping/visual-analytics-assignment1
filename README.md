# visual-analytics-assignment1
First assignment for visual analytics course.
This assignment is oriented at image retrieval in the [17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

## Setup

After downloading the data all jpg files should be arranged in one folder.
In the examples I will use the following folder structure:

```
 - data/
    - jpg/
        - files.txt
        - image_0001.jpg
        ...
```

The code ignore the `files.txt` file and scans the directory for all images.
This choice was made so that the code is easily reusable in non-indexed image datasets.

Each image is assumed to have a particular ID, this is the stem of the image path.
E.g. for the file `image_0001.jpg` its ID would be `"image_0001"`.
These IDs will be later used for specifying which image to base the search on.


To run the code first install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The repository contains code for a command line interface that can search in the dataset based on one user-specified image.
The CLI can use multiple distance metrics and types of latent representations to achieve this.

### Color Histograms

To do image retrieval based on minmax normalized color histograms, run the following:

```bash
python3 src/hist_search.py data/jpg/image_0001.jpg -o "out/hist"
```

This will put a CSV file with the images closest to the target by Chi Square histogram distance in the `out/hist` folder.

```
 - out/
    - hist/
        - image_0001.csv
```

These are the results I got:

| |Filename|Distance|
|-|-|-|
| 0|image_0001.jpg|0.0|
| 1|image_0773.jpg|190.13992491162105|
| 2|image_1316.jpg|190.2249241130487|
| 3|image_0740.jpg|190.62783760197846|
| 4|image_1078.jpg|191.69055452774253|
| 5|image_0319.jpg|191.8753821638015|

### VGG16 

To use VGG16 image embeddings and cosine distance to search in the images run this command:

```bash
python3 src/embedding_search.py data/jpg/image_0001.jpg -o "out/vgg16"
```

This will put a CSV file with the images closest to the target cosine distance in the `out/vgg16` folder.

```
 - out/
    - vgg16/
        - image_0001.csv
```

### Parameters

| Argument                | Description                                                                                  | Type    | Default           |
|-------------------------|----------------------------------------------------------------------------------------------|---------|-------------------|
| `image_id`              | Identifier for the image.                                                                    |         | -                 |
| `-h`, `--help`          | Show this help message and exit.                                                             |         |                   |
| `-i IMAGES_PATH`,<br>`--images_path IMAGES_PATH` | Path to the source directory containing images.                                           | str     | `"data/jpg"`                 |
| `-o OUT_PATH`,<br>`--out_path OUT_PATH` | Path to the output directory to save results.                                                | str     | `"out"`                 |
| `-k TOP_K`,<br>`--top_k TOP_K` | Top K similar images to retrieve.                                                             | int     | 5                 |

