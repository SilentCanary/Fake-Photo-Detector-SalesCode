# Fake-Photo-Detector-SalesCode

live link :- https://fake-photo-detector-salescode.onrender.com

Note: On Render free tier, the first request may be slower because of cold start. Local benchmark latency is the reported inference latency.


# Spot the Fake Photo

A small computer-vision pipeline for detecting whether an input image is a **real photo** or a **photo of another screen / printout** (recapture).

The final predictor follows the required interface:

```bash
python predict.py some_image.jpg
```

It prints a single score from `0` to `1`:

- `0` = likely real photo
- `1` = likely screen recapture

Example:

```bash
python predict.py data/screen/example.jpeg
```

Output:

```text
0.8732
```

---

## Approach

This solution uses a lightweight handcrafted feature pipeline plus Logistic Regression.

Instead of recognizing the object in the image, the model looks for physical artifacts commonly introduced when a camera photographs another screen or printout, such as:

- moiré / periodic frequency patterns using FFT features
- local texture changes using LBP features
- glare, saturation, and brightness statistics in HSV space
- RGB channel mismatch / chromatic edge behavior
- edge density, blur, and sharpness statistics

The extracted features are passed to a small Logistic Regression classifier, which outputs a recapture score.

---

## Project Structure

```text
spot_fake_photo/
│
├── data/
│   ├── real/                  # real photos, label 0
│   └── screen/                # recaptured screen/printout photos, label 1
│
├── src/
│   ├── build_features_csv.py  # extracts features into a CSV
│   ├── train.py               # trains Logistic Regression with 5-fold CV
│   └── utils/
│       ├── image_io.py
│       ├── fft_features.py
│       ├── lbp_features.py
│       ├── color_features.py
│       ├── rgb_phase_features.py
│       ├── edge_features.py
│       └── feature_orchestrator.py
│
├── models/
│   └── recapture_logistic.joblib
│
├── outputs/
│   ├── features.csv
│   └── cv_results.txt
│
├── predict.py                 # required one-line predictor
├── benchmark.py               # latency benchmark
├── app.py                     # optional Gradio live demo
├── requirements.txt
└── README.md
```

---

## Setup

Create and activate a Python environment:

```bash
conda create -n salescode_task python=3.11
conda activate salescode_task
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Layout

Place images in this structure:

```text
data/
  real/
    real_image_1.jpg
    real_image_2.jpg
    ...

  screen/
    recapture_image_1.jpg
    recapture_image_2.jpg
    ...
```

The code assigns labels using folder names:

- `data/real/` → label `0`
- `data/screen/` → label `1`

Image filenames do not matter.

---

## Build Features

Run from the project root:

```bash
python -m src.build_features_csv --data_dir data --out outputs/features.csv
```

This creates:

```text
outputs/features.csv
```

Each image becomes one row of handcrafted features.

---

## Train Model

Train Logistic Regression with 5-fold stratified cross-validation:

```bash
python -m src.train --features outputs/features.csv --model_out models/recapture_logistic.joblib --report_out outputs/cv_results.txt
```

This creates:

```text
models/recapture_logistic.joblib
outputs/cv_results.txt
```

The final model is retrained on the full available dataset after cross-validation.

---

## Current Validation Result

On the current self-collected dataset of 135 images:

```text
Real images:   63
Screen images: 72
```

5-fold stratified cross-validation gave:

```text
Out-of-fold accuracy: 96.30%
Out-of-fold F1:       96.50%
Out-of-fold ROC-AUC:  98.81%
Best threshold:       0.43
```

Confusion matrix at the best threshold:

```text
[[61  2]
 [ 3 69]]
```

This means:

- 61 real photos correctly classified as real
- 2 real photos incorrectly flagged as screen
- 69 screen recaptures correctly classified as screen
- 3 screen recaptures missed as real

These numbers are also saved in:

```text
outputs/cv_results.txt
```

---

## Predict One Image

Required command:

```bash
python predict.py some_image.jpg
```

Example:

```bash
python predict.py data/screen/example.jpeg
```

Output:

```text
0.8732
```

The script prints only one number, as required.

---

## Benchmark Latency

Run:

```bash
python benchmark.py --input data --model models/recapture_logistic.joblib --repeat 3
```

Current measured latency on a Windows laptop CPU:

```text
Mean latency:   106.62 ms/image
Median latency: 110.34 ms/image
P95 latency:    137.58 ms/image
```

This was measured over 135 images using `benchmark.py`.

---

## Cost Per Image

The model runs locally using OpenCV feature extraction and a small Logistic Regression model.

```text
Cloud cost per image: $0
```

For mobile deployment, the same idea can run on-device. The only cost is local CPU time.

---

## Optional Live Demo

A small Gradio demo is included.

Run locally:

```bash
python app.py
```

Then open the displayed local URL, usually:

```text
http://127.0.0.1:7860
```

The app supports:

- image upload
- webcam capture
- recapture score display
- predicted label
- latency display

---

## Design Trade-offs

This project intentionally uses a small classical computer-vision pipeline instead of a large deep learning model.

Advantages:

- explainable features
- small model file
- no cloud API
- low cost
- CPU-only inference
- works with limited training data

Limitations:

- dataset is still small
- current data comes from limited devices and environments
- hidden-test performance may vary with different phones, screens, lighting, and recapture styles
- Python/OpenCV prototype is slower than a native mobile implementation

---

## Future Improvements

With more time/data, I would improve the system by:

1. Collecting more hard examples from different phones, monitors, lighting conditions, and camera qualities.
2. Adding hard real negatives such as real laptops, monitors, shiny objects, glare, and low-light photos.
3. Testing on external recapture / moiré datasets where license permits.
4. Running feature ablation to remove slow or low-value features.
5. Comparing this pipeline with a quantized MobileNetV3-Small / TFLite model for faster mobile deployment.
6. Periodically retraining with newly observed failure cases as cheaters adapt.

---

## Quick Command Summary

```bash
conda activate brown_task
pip install -r requirements.txt

python -m src.build_features_csv --data_dir data --out outputs/features.csv

python -m src.train --features outputs/features.csv --model_out models/recapture_logistic.joblib --report_out outputs/cv_results.txt

python predict.py data/screen/example.jpeg

python benchmark.py --input data --model models/recapture_logistic.joblib --repeat 3

python app.py
```

## Author 👤
SilentCanary (Advitiya)
