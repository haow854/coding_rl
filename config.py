# ── Base models (HuggingFace) ────────────────────────────────────
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"

# ── Trained model IDs (fill in after each upload) ────────────────
TRAINED_PLANNER_ID = ""   # e.g. "your_hf_name/Planner_3B_1.0"
TRAINED_CODER_ID   = ""   # e.g. "your_hf_name/Coder_7B_1.0"
TRAINED_OPT_ID     = ""   # e.g. "your_hf_name/Optimizer_7B_1.0"

# ── HuggingFace upload ───────────────────────────────────────────
HF_ACCOUNT = ""
HF_TOKEN   = ""

# ── Upload: set this before running upload.py ────────────────────
UPLOAD_MODEL_PATH  = ""     # local model path to upload
UPLOAD_MODEL_NAME = ""    # e.g. "Planner_3B_1.0"