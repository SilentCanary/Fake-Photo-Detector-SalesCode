# Short Note — Spot the Fake Photo

## Approach

I built a lightweight computer-vision pipeline to classify whether an input image is a direct real-world photo or a recaptured photo of another screen/printout. Instead of recognizing objects, the system extracts artifacts that are common in screen recaptures: moiré/frequency spikes, repeated micro-texture, glare and saturation behavior, RGB-channel mismatch, edge density, and blur/sharpness statistics.

Each image is resized to 256×256 and converted into handcrafted OpenCV features. The main feature groups are FFT-based frequency features, Local Binary Pattern texture features, HSV color/glare features, RGB channel-gradient features, and edge/sharpness features. These features are passed to a small Logistic Regression classifier, which outputs a score from 0 to 1, where 0 means real photo and 1 means screen recapture.

I chose Logistic Regression because it was small, fast, explainable, and performed better than Decision Tree and Random Forest on my validation experiments.

## Accuracy

I collected a small self-captured dataset with 135 images:

- 63 real photos
- 72 screen/recapture photos

Using 5-fold stratified cross-validation, the final Logistic Regression model achieved:

- Out-of-fold accuracy: 96.30%
- Out-of-fold ROC-AUC: 0.9881
- Out-of-fold F1 score: 0.9650

The confusion matrix at the selected threshold was:

```text
[[61  2]
 [ 3 69]]
```

So the model correctly classified 61/63 real photos and 69/72 screen recaptures in out-of-fold evaluation.

## Latency and Cost

Latency was measured using `benchmark.py` over 135 images on my Windows laptop CPU:

Latency results
--------------------------------------------------
Total measured predictions: 405
Mean latency:   71.79 ms/image
Median latency: 71.80 ms/image
P95 latency:    82.57 ms/image
Min latency:    52.12 ms/image
Max latency:    94.57 ms/image

Cost per image is effectively $0 cloud cost because the model runs locally/on-device using OpenCV feature extraction and a small scikit-learn model. No external API or GPU server is required.

## What I would improve with more time

The biggest improvement would be collecting more diverse data across more phones, screens, lighting conditions, and display types. I would especially add hard real negatives such as real laptop/phone screens, shiny objects, glare, mirrors, and low-light photos, plus hard recaptures such as fullscreen screen photos with no visible border.

To keep the system accurate as cheaters adapt, I would continuously collect failed/edge cases and retrain the classifier periodically. I would also compare this handcrafted pipeline with a quantized MobileNetV3-Small model or an ensemble of both scores if a larger dataset becomes available.

For phone deployment, I would move the feature extraction to native code, keep the 256×256 resize, remove non-contributing features through ablation, and package the classifier in a lightweight on-device format.

The cut-off threshold was selected using out-of-fold cross-validation predictions. The chosen threshold was 0.43, which balanced accuracy and recall on my validation data.
