import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from addition_constants import HF_TOKEN, MODEL_NAME


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN


def load_gemma_model(device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR)
    model.eval()
    return model, tokenizer
