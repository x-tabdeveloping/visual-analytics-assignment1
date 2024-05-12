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
python3 src/image_search.py image_0001 -o "out/hist" -r "hist" -d "chisqr"
```

This will put a CSV file with the images closest to the target by Chi Square histogram distance in the `out/hist` folder.

```
 - out/
    - hist/
        - image_0001.csv
```

These are the results I got:

|   | Filename               | Distance            |
|---|------------------------|---------------------|
| 0 | Target                 | 0.0                 |
| 1 | data/jpg/image_0718.jpg | 15.246157334036242 |
| 2 | data/jpg/image_1010.jpg | 20.437279080399684 |
| 3 | data/jpg/image_0593.jpg | 20.51119057774688  |
| 4 | data/jpg/image_0686.jpg | 22.747755263996137 |
| 5 | data/jpg/image_0054.jpg | 23.153972570524715 |

### VGG16 

To use VGG16 image embeddings to search in the images run this command:

 > Remember to set the distance metric to cosine

```bash
python3 src/image_search.py image_0001 -o "out/vgg16" -r "vgg16" -d "cosine"
```

### Parameters

| Argument                | Description                                                                                  | Type    | Default           |
|-------------------------|----------------------------------------------------------------------------------------------|---------|-------------------|
| `image_id`              | Identifier for the image.                                                                    |         | -                 |
| `-h`, `--help`          | Show this help message and exit.                                                             |         |                   |
| `-s SOURCE_PATH`,<br>`--source_path SOURCE_PATH` | Path to the source directory containing images.                                           | str     | `"data/jpg"`                 |
| `-o OUT_PATH`,<br>`--out_path OUT_PATH` | Path to the output directory to save results.                                                | str     | `"out"`                 |
| `-r REPRESENTATION`,<br>`--representation REPRESENTATION` | Type of representation to use for image features. (VGG16 or minmax normalized color histograms)                                         | `{"hist", "vgg16"}`     | `"hist"`                 |
| `-k TOP_K`,<br>`--top_k TOP_K` | Top K similar images to retrieve.                                                             | int     | 5                 |
| `-d DISTANCE_METRIC`,<br>`--distance_metric DISTANCE_METRIC` | Distance metric to use for similarity calculation. (Cosine or Chi square histogram distance)                                        | `{"cosine", "chisqr"}`     | `"chisqr"`                 |

