import os
import hashlib
import numpy as np
import onnxruntime as ort
import gradio as gr
from PIL import Image

# ============================================================================
# CONFIG — confirmed class order (alphabetical, matches training exactly)
# ============================================================================
CLASS_NAMES = [
    'AMD', 'Cataract', 'Diabetic Retinopathy', 'Glaucoma',
    'Hypertension', 'Myopia', 'Normal', 'Others'
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_CONFIGS = {
    "General-Purpose (ResNet-50)": {
        "path": "models/resnet50_m5a.onnx",
        "input_size": 224,
        "blurb": (
            "Best overall diagnostic accuracy across all 8 classes "
            "(71.7% test accuracy, Macro-F1 0.60). Lower sensitivity "
            "to rare conditions like Hypertension (15% recall)."
        ),
    },
    "High-Sensitivity (EfficientNet-B5)": {
        "path": "models/efficientnet_b5_m6.onnx",
        "input_size": 300,
        "blurb": (
            "Prioritizes catching rare, high-risk conditions — Hypertension "
            "recall 38.5% (vs 15.4% for the general-purpose model) — at the "
            "cost of lower overall accuracy (62.6%) and reduced precision "
            "on Diabetic Retinopathy / Normal."
        ),
    },
}

# ============================================================================
# MODEL LOADING — lazy, cached per session (CPU-only, no GPU needed)
# ============================================================================
_loaded_sessions = {}

def get_session(model_key):
    if model_key not in _loaded_sessions:
        model_path = MODEL_CONFIGS[model_key]["path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Ensure both .onnx files are present under models/."
            )
        _loaded_sessions[model_key] = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
    return _loaded_sessions[model_key]

# ============================================================================
# RESULT CACHE — keyed by (image content hash, model choice)
# ============================================================================
_result_cache = {}

def _hash_image(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()

# ============================================================================
# PREPROCESSING — per-model resolution (224 for ResNet-50, 300 for EfficientNet-B5)
# ============================================================================
def preprocess(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)  # add batch dim
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ============================================================================
# CORE PREDICTION FUNCTION (the single public endpoint)
# ============================================================================
def classify_retinal_image(image: Image.Image, model_choice: str):
    if image is None:
        raise gr.Error("Please upload a fundus image before running the classifier.")

    cache_key = (_hash_image(image), model_choice)
    if cache_key in _result_cache:
        return _result_cache[cache_key]

    config = MODEL_CONFIGS[model_choice]
    session = get_session(model_choice)
    input_tensor = preprocess(image, config["input_size"])

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor})[0][0]
    probs = softmax(logits)

    result = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    _result_cache[cache_key] = result
    return result

# ============================================================================
# EXAMPLES — verified discovery (only list files that actually exist on disk)
# ============================================================================
EXAMPLES_DIR = "assets/samples"

def discover_examples():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    valid_ext = ('.jpg', '.jpeg', '.png')
    return [
        [os.path.join(EXAMPLES_DIR, f)]
        for f in sorted(os.listdir(EXAMPLES_DIR))
        if f.lower().endswith(valid_ext)
    ]

EXAMPLE_LIST = discover_examples()

# ============================================================================
# "QUANTUM BLACK" DESIGN SYSTEM — deep black, cyan/violet accents, eye-comfortable
# ============================================================================
QUANTUM_BLACK_CSS = """
:root {
    --qb-bg: #08080c;
    --qb-panel: #101017;
    --qb-panel-alt: #15151e;
    --qb-border: #26263a;
    --qb-text: #d6d6e0;
    --qb-text-dim: #9494a8;
    --qb-accent: #4fd8ff;
    --qb-accent-2: #9b6bff;
    --qb-warn-bg: #1c1408;
    --qb-warn-border: #4a3812;
    --qb-warn-text: #e8c073;
}

.gradio-container {
    background: radial-gradient(circle at 15% 0%, #0d0d17 0%, var(--qb-bg) 45%) !important;
    color: var(--qb-text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

#qb-header {
    text-align: left;
    padding: 4px 0 2px 0;
}
#qb-header h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, var(--qb-accent), var(--qb-accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px !important;
}

#qb-tradeoff {
    background: var(--qb-warn-bg) !important;
    border: 1px solid var(--qb-warn-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 14px !important;
}
#qb-tradeoff h3 { color: var(--qb-warn-text) !important; margin-top: 0 !important; }
#qb-tradeoff p, #qb-tradeoff li { color: var(--qb-text-dim) !important; line-height: 1.55; }
#qb-tradeoff strong { color: var(--qb-text) !important; }

.qb-card {
    background: var(--qb-panel) !important;
    border: 1px solid var(--qb-border) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    box-shadow: 0 0 0 1px rgba(79, 216, 255, 0.03), 0 8px 24px rgba(0,0,0,0.35) !important;
}

#qb-upload {
    background: var(--qb-panel) !important;
    border: 1px dashed var(--qb-border) !important;
    border-radius: 16px !important;
}

#qb-model-selector .wrap {
    background: var(--qb-panel-alt) !important;
    border: 1px solid var(--qb-border) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
#qb-model-selector label {
    border-radius: 9px !important;
    color: var(--qb-text-dim) !important;
    transition: all 0.15s ease-in-out;
}
#qb-model-selector input:checked + label,
#qb-model-selector label.selected {
    background: linear-gradient(90deg, rgba(79,216,255,0.15), rgba(155,107,255,0.15)) !important;
    color: var(--qb-accent) !important;
    box-shadow: inset 0 0 0 1px rgba(79,216,255,0.35) !important;
}

#qb-blurb {
    color: var(--qb-text-dim) !important;
    font-size: 0.92rem !important;
    padding: 4px 2px 8px 2px !important;
    border-left: 2px solid var(--qb-accent) !important;
    padding-left: 10px !important;
}

#qb-submit {
    background: linear-gradient(90deg, var(--qb-accent) 0%, var(--qb-accent-2) 100%) !important;
    border: none !important;
    color: #050508 !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(79, 216, 255, 0.25) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
#qb-submit:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 26px rgba(155, 107, 255, 0.35) !important;
}

#qb-results {
    background: var(--qb-panel) !important;
    border: 1px solid var(--qb-border) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}

#qb-footer {
    text-align: center;
    color: var(--qb-text-dim) !important;
    font-size: 0.8rem !important;
    padding-top: 10px !important;
    opacity: 0.7;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--qb-bg); }
::-webkit-scrollbar-thumb { background: var(--qb-border); border-radius: 8px; }
"""

QUANTUM_BLACK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#08080c",
    background_fill_primary="#101017",
    background_fill_secondary="#15151e",
    border_color_primary="#26263a",
    block_background_fill="#101017",
    block_border_color="#26263a",
    block_label_text_color="#9494a8",
    body_text_color="#d6d6e0",
    body_text_color_subdued="#9494a8",
)

TRADE_OFF_NOTE = """
### ⚠️ Model Trade-Off — Read Before Interpreting Results
This project trains two valid "champion" models rather than picking one and hiding the trade-off:

- **General-Purpose (ResNet-50)** — best overall accuracy across all 8 classes.
- **High-Sensitivity (EfficientNet-B5)** — dramatically better at catching rare Hypertension cases (38.5% vs 15.4% recall), at the cost of overall accuracy and precision on more common classes.

Neither model is "correct" in isolation — the right choice depends on whether missing a rare, high-risk case (favor High-Sensitivity) or overall diagnostic reliability (favor General-Purpose) matters more for your use case. This demo is for research/educational purposes only and is **not a certified diagnostic tool**.
"""

with gr.Blocks(theme=QUANTUM_BLACK_THEME, css=QUANTUM_BLACK_CSS, title="Retinal Disease Classifier") as demo:

    with gr.Column(elem_id="qb-header"):
        gr.Markdown("# 👁️ Retinal Fundus Disease Classifier")

    with gr.Column(elem_id="qb-tradeoff"):
        gr.Markdown(TRADE_OFF_NOTE)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Fundus Image", elem_id="qb-upload")

            with gr.Column(elem_classes=["qb-card"]):
                model_selector = gr.Radio(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="General-Purpose (ResNet-50)",
                    label="Select Model",
                    elem_id="qb-model-selector"
                )
                model_info = gr.Markdown(
                    MODEL_CONFIGS["General-Purpose (ResNet-50)"]["blurb"],
                    elem_id="qb-blurb"
                )
                submit_btn = gr.Button("Classify", variant="primary", elem_id="qb-submit")

            if EXAMPLE_LIST:
                gr.Examples(
                    examples=EXAMPLE_LIST,
                    inputs=image_input,
                    label="Try a sample fundus image"
                )

        with gr.Column(scale=1):
            with gr.Column(elem_id="qb-results"):
                output_label = gr.Label(num_top_classes=8, label="Predicted Class Probabilities")

    gr.Markdown(
        "Research/educational demo — not a certified medical diagnostic tool.",
        elem_id="qb-footer"
    )

    def update_blurb(model_choice):
        return MODEL_CONFIGS[model_choice]["blurb"]

    model_selector.change(fn=update_blurb, inputs=model_selector, outputs=model_info, api_name=False)

    submit_btn.click(
        fn=classify_retinal_image,
        inputs=[image_input, model_selector],
        outputs=output_label,
        api_name="classify"  # only this endpoint is exposed publicly / to MCP
    )

if __name__ == "__main__":
    import sys, traceback
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            ssr_mode=False,
            show_api=False,
            mcp_server=True
        )
    except Exception as e:
        print("[MCP LAUNCH ERROR]:", repr(e), flush=True)
        traceback.print_exc()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            ssr_mode=False,
            show_api=False
        )