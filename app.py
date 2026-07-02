from pathlib import Path
import tempfile
import time

import gradio as gr
import numpy as np
from PIL import Image

from predict import predict
from predict import load_model_bundle

MODEL_PATH = "models/recapture_logistic.joblib"
THRESHOLD = 0.59
try:
    _bundle = load_model_bundle(MODEL_PATH)
    THRESHOLD = float(_bundle.get("threshold", 0.59))
    print(f"Model preloaded successfully. Threshold={THRESHOLD:.2f}", flush=True)
except Exception as e:
    print(f"Model preload failed: {e}", flush=True)

def predict_from_image(image):
    if image is None:
        return None, "Please upload or capture an image.", ""

    start = time.perf_counter()

    # Gradio gives PIL image / numpy image depending on version
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        image.save(temp_path, format="JPEG", quality=95)

    score = predict(temp_path, MODEL_PATH)

    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0

    Path(temp_path).unlink(missing_ok=True)

    if score >= THRESHOLD:
        label = "SCREEN / RECAPTURE"
        explanation = "This image looks like a photo of another screen or printout."
    else:
        label = "REAL PHOTO"
        explanation = "This image looks like a direct real-world photo."

    result_text = (
        f"Prediction: {label}\n"
        f"Score: {score:.4f}\n"
        f"Threshold: {THRESHOLD:.2f}"
    )

    latency_text = f"Latency: {latency_ms:.2f} ms/image"

    return score, result_text, latency_text + "\n" + explanation


with gr.Blocks(title="Spot the Fake Photo") as demo:
    gr.Markdown(
        """
        # Spot the Fake Photo
        
        Detect whether an image is a **real photo** or a **photo of another screen/printout**.
        
        Score meaning:
        - `0.0` = likely real photo
        - `1.0` = likely screen recapture
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload image or use webcam",
                sources=["upload", "webcam"],
                type="pil",
            )

            predict_button = gr.Button("Predict")

        with gr.Column():
            score_output = gr.Number(
                label="Recapture Score",
                precision=4,
            )

            result_output = gr.Textbox(
                label="Result",
                lines=4,
            )

            details_output = gr.Textbox(
                label="Latency / Details",
                lines=4,
            )

    predict_button.click(
        fn=predict_from_image,
        inputs=image_input,
        outputs=[score_output, result_output, details_output],
    )

    gr.Markdown(
        """
        ### Method
        This demo uses handcrafted computer-vision features:
        FFT/moiré statistics, native-resolution patch FFT/residual features, LBP texture, HSV glare/saturation, RGB-channel behavior, and edge/sharpness features.
        A small Logistic Regression model converts these features into a recapture score.
        """
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
