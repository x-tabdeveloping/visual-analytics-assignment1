# visual-analytics-assignment1
First assignment for visual analytics course.

I have chosen the first image as the target (`image_0001.csv`).
The code expects all pictures to be in the `data/jpg` folder.

To run the code first install dependencies:
```bash
pip install pillow tqdm pandas numpy
```

Then run the script:
```bash
python3 analysis.py
```

This will put a CSV file with the images closest to the target by Chi Square histogram distance in the `out/` folder.

These are the results I got:

| Filename                  | Distance                  |
|---------------------------|---------------------------|
| Target                    | 0.0                       |
| data/jpg/image_0912.jpg   | 189340.51655981614        |
| data/jpg/image_0718.jpg   | 215865.7637805672         |
| data/jpg/image_0149.jpg   | 230148.24425520096        |
| data/jpg/image_1010.jpg   | 239003.0483289215         |
| data/jpg/image_0348.jpg   | 240056.94795960546        |
