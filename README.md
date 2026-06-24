# Face Swap

Replace a **specific person** in a video with a face of your choice. The script tracks the target identity across every frame using face embeddings, so it follows the person even as they turn, move, or change expression.

![UI Screenshot](https://img.shields.io/badge/UI-Gradio-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## How it works

1. You upload a **source photo** (the face you want to paste).
2. You upload a **target video** (the video you want to edit).
3. The script detects all faces in the first frame of the video. You pick **which person** to replace.
4. For every subsequent frame it:
   - Detects all faces.
   - Finds the one that matches your chosen person (by **embedding similarity**).
   - Swaps the face using [InsightFace](https://github.com/deepinsight/insightface)'s `inswapper_128` model.
5. The result is saved as an MP4 video.

---

## Quick start

### 1. Install

```bash
# Clone
git clone https://github.com/vinothvikas1987/face-swap.git
cd face-swap

# Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `onnxruntime-gpu` requires a CUDA-capable GPU with the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) installed. If you don't have a GPU, replace it with `onnxruntime` in `requirements.txt` — the script will fall back to CPU automatically (much slower but works).

### 2. Run

```bash
python app.py
```

The model downloads automatically on the first run (~170 MB). A Gradio UI opens in your browser.

### 3. Use

1. **Source** – Upload a photo with a clear, front-facing shot of the person you want to paste.
2. **Target** – Upload the video you want to edit.
3. **Face #** – If multiple faces are detected, pick which one to use (0 = first from left).
4. **Tracking sensitivity** – Controls how strictly the script matches the target person:
   - **0.4–0.5** – Very strict (fewer swaps, low false-positive rate).
   - **0.6** (default) – Good balance.
   - **0.8–1.0** – Loose (more swaps, higher chance of swapping the wrong person).
5. Click **Run Swap**.

---

## Where it works best

| Scenario | Result |
|---|---|
| One person in frame, front-facing source photo | Excellent |
| Person turns their head (profile / ¾ view) | Good — embedding averaging helps |
| Multiple people in the video | Good — you pick which identity to track |
| High-resolution video (720p+) | Best — face detection is more reliable |
| Same lighting between source and target | Best — colour mismatch is minimised |
| Source photo is high-quality, front-facing | Best — the swap looks natural |

---

## Limitations

| Issue | Why | Mitigation |
|---|---|---|
| **No GPU** | ONNX inference on CPU is 10–50× slower | Use a GPU, reduce video resolution, or swap to `onnxruntime` (CPU) and accept longer runtimes |
| **Fast head turns / motion blur** | Face detection may miss the target in some frames | Lower tracking sensitivity slightly (0.5–0.6) |
| **Extreme angles** (top-down, behind head) | No face visible → nothing to swap | Unavoidable — the model needs a visible face |
| **Colour mismatch** | Source and target may have different lighting/skin tones | Colour-correction is not applied (you can add it externally with e.g. `moviepy`) |
| **Large age / style difference** | The model works but the result may look unnatural | Best results when source and target are similar in age, gender, and face shape |
| **Occlusions** (hand in front of face, sunglasses, mask) | The swap will be applied to the visible face area only | Works partially; heavy occlusions may degrade quality |
| **Very low resolution** (< 360p) | Face landmarks are inaccurate | Upscale the video first (e.g. with `ffmpeg`) |

---

## Model

This project uses **InsightFace's `inswapper_128`**, a lightweight ONNX model purpose-built for real-time face swapping. It is downloaded automatically from Hugging Face on the first run.

- **Model:** [`inswapper_128.onnx`](https://huggingface.co/ezioruan/inswapper_128.onnx)
- **Size:** ~170 MB
- **Paper:** [InsightFace: 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface)

---

## Project structure

```
face-swap/
├── app.py              ← Main inference script (Gradio UI)
├── requirements.txt    ← Python dependencies
├── .gitignore
├── LICENSE             ← MIT
├── README.md
├── models/             ← Models downloaded here on first run
└── output/             ← Rendered videos saved here
```

---

## License

[MIT](LICENSE) — do what you want, but don't blame us if the swap looks creepy.
