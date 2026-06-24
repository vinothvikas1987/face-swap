"""
Face Swap — Identity-Locked Video Face Swapping

Replaces a specific person in a video with a source face,
tracking the target identity across frames using face embeddings.

Run with:  python app.py
"""

import os
import sys

# ── Monkey-patch: ensure HfFolder exists for Gradio's oauth module ──
# Older Gradio versions import `HfFolder` from `huggingface_hub`, but
# newer releases of the library have removed it.  We shim it so the
# import does not crash (mostly relevant on Hugging Face Spaces).
try:
    import huggingface_hub as _hf
    if not hasattr(_hf, "HfFolder"):
        class _HfFolderShim:
            @staticmethod
            def get_token():
                return os.environ.get("HF_TOKEN")
            @staticmethod
            def save_token(token: str) -> None:
                pass
            @staticmethod
            def delete_token() -> None:
                pass
        _hf.HfFolder = _HfFolderShim
        sys.modules["huggingface_hub.hf_folder"] = type(sys)("hf_folder")
        sys.modules["huggingface_hub.hf_folder"].HfFolder = _HfFolderShim
except ImportError:
    pass

import cv2
import numpy as np
import gradio as gr
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Model handling — auto-download from Hugging Face
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "models"
MODEL_URLS = {
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
}


def ensure_model(model_name: str) -> Path:
    """Download the model if it does not exist locally."""
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        return model_path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = MODEL_URLS[model_name]
    print(f"Downloading {model_name} …")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {100 * downloaded // total}%", end="", flush=True)
    print(f"\nDownloaded {model_name}")
    return model_path


# ---------------------------------------------------------------------------
# Provider detection (CUDA → CPU fallback)
# ---------------------------------------------------------------------------

def pick_providers():
    """Return ONNX Runtime providers: CUDA if available, otherwise CPU."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return [
                ("CUDAExecutionProvider", {"device_id": 0, "gpu_mem_limit": 4 * 1024 * 1024 * 1024}),
                "CPUExecutionProvider",
            ]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Lazy model initialisation
# ---------------------------------------------------------------------------

_face_app = None
_swapper = None


def get_models():
    global _face_app, _swapper
    if _face_app is not None and _swapper is not None:
        return _face_app, _swapper

    providers = pick_providers()
    print(f"Using providers: {providers}")

    from insightface.app import FaceAnalysis
    import insightface

    _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    _face_app.prepare(ctx_id=0, det_size=(640, 640))

    model_path = str(ensure_model("inswapper_128.onnx"))
    _swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    print("Models loaded successfully.")
    return _face_app, _swapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_faces(img):
    """Detect faces sorted left-to-right."""
    if img is None:
        return None, []
    faces = get_models()[0].get(img)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    previews = []
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = img[max(0, y1):y2, max(0, x1):x2]
        previews.append((crop, f"Face #{i}"))
    return faces, previews


def get_faces_from_video(video_path):
    """Detect faces in the first frame of a video."""
    if video_path is None:
        return None, []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, []
    return get_faces(frame)


def match_face(faces, target_embedding, threshold):
    """Return the face closest to *target_embedding* within *threshold*."""
    best = None
    best_dist = threshold
    for face in faces:
        dist = np.linalg.norm(face.normed_embedding - target_embedding)
        if dist < best_dist:
            best_dist = dist
            best = face
    return best


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def run_swap(source_img, target_video, source_idx, target_idx,
             sim_thresh, progress=gr.Progress()):
    """
    Main pipeline:
      1. Pick the source face.
      2. Lock onto the target face (first frame).
      3. Track & swap frame-by-frame.
    """
    if source_img is None or target_video is None:
        return "Please provide both a source image and a target video.", None

    face_app, swapper = get_models()

    # --- source face ---
    src_faces = face_app.get(source_img)
    src_faces = sorted(src_faces, key=lambda x: x.bbox[0])
    if not src_faces:
        return "No face detected in the source image.", None
    src_idx = int(source_idx)
    if src_idx >= len(src_faces):
        return f"Source face #{src_idx} not found (only {len(src_faces)} detected).", None
    src_face = src_faces[src_idx]

    # --- target identity (lock-on from frame 1) ---
    cap = cv2.VideoCapture(target_video)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return "Could not read the target video.", None

    tgt_faces = face_app.get(first_frame)
    tgt_faces = sorted(tgt_faces, key=lambda x: x.bbox[0])
    tgt_idx = int(target_idx)
    if tgt_idx >= len(tgt_faces):
        cap.release()
        return f"Target face #{tgt_idx} not found in frame 1.", None
    target_embedding = tgt_faces[tgt_idx].normed_embedding

    # --- video loop ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    swaps = 0

    for i in progress.tqdm(range(total), desc="Swapping"):
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_app.get(frame)
        match = match_face(faces, target_embedding, sim_thresh)
        if match is not None:
            frame = swapper.get(frame, match, src_face, paste_back=True)
            swaps += 1
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return f"Done — {swaps}/{total} frames swapped. Saved to {output_path}", output_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

css = """
footer {visibility: hidden}
.app-title {text-align: center}
"""

with gr.Blocks(css=css, title="Face Swap") as ui:
    gr.Markdown(
        "# Face Swap\n"
        "Replace a specific person in a video with a face of your choice. "
        "The script **tracks the target identity** across the whole video using face embeddings."
    )

    src_state = gr.State([])
    tgt_state = gr.State([])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Source (the face you want to paste)")
            src_img = gr.Image(label="Upload a photo", type="numpy")
            src_gallery = gr.Gallery(label="Detected faces", columns=3, height=150)
            src_id = gr.Number(label="Face # to use", value=0, precision=0)
        with gr.Column():
            gr.Markdown("### 2. Target (the video to edit)")
            tgt_vid = gr.Video(label="Upload a video")
            tgt_gallery = gr.Gallery(label="Faces found in first frame", columns=3, height=150)
            tgt_id = gr.Number(label="Person # to replace", value=0, precision=0)

    with gr.Row():
        thresh = gr.Slider(
            minimum=0.1, maximum=1.5, value=0.6, step=0.05,
            label="Tracking sensitivity (lower = stricter match)"
        )
        run_btn = gr.Button("Run Swap", variant="primary", scale=1)

    log = gr.Textbox(label="Result")
    preview = gr.Video(label="Output")

    # Wire UI events
    src_img.change(fn=get_faces, inputs=src_img, outputs=[src_state, src_gallery])
    tgt_vid.change(fn=get_faces_from_video, inputs=tgt_vid, outputs=[tgt_state, tgt_gallery])
    run_btn.click(
        fn=run_swap,
        inputs=[src_img, tgt_vid, src_id, tgt_id, thresh],
        outputs=[log, preview],
    )

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
