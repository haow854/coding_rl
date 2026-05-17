from config import UPLOAD_MODEL_PATH, UPLOAD_MODEL_NAME, HF_ACCOUNT, HF_TOKEN
import torch
from unsloth import FastLanguageModel

peft_model, peft_tokenizer = FastLanguageModel.from_pretrained(
    model_name=UPLOAD_MODEL_PATH,
    max_seq_length=16384,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)


peft_model.push_to_hub_merged(
    f"{HF_ACCOUNT}/{UPLOAD_MODEL_NAME}", peft_tokenizer, token=HF_TOKEN
)