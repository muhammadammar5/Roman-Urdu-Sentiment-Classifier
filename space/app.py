"""Gradio demo for the Roman Urdu Sentiment Classifier (Hugging Face Space)."""
import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from normalize import normalize_text

MODEL_ID = os.getenv("MODEL_ID", "muhammadammar5/roman-urdu-sentiment-distilbert")
GITHUB_URL = "https://github.com/muhammadammar5/Roman-Urdu-Sentiment-Classifier"

try:
    CONFIG = json.loads(open(hf_hub_download(MODEL_ID, "rus_config.json"), encoding="utf-8").read())
except Exception:
    CONFIG = {}
NORMALIZE = CONFIG.get("normalize", True)
MAX_LEN = CONFIG.get("max_len", 64)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}

EXAMPLES = [
    "Yeh phone bohat acha hai, battery zabardast hai",
    "Bakwas product, paisa zaya ho gaya",
    "Delivery kal tak aa jaye gi",
    "Service bilkul kharab thi, kabhi dobara nahi lunga",
    "Mujhe yeh dress bohat pasand aayi, shukriya!",
]


@torch.inference_mode()
def classify(text: str):
    if not text or not text.strip():
        return {}
    cleaned = normalize_text(text) if NORMALIZE else text.strip()
    enc = tokenizer(cleaned, truncation=True, max_length=MAX_LEN, return_tensors="pt",
                    return_token_type_ids=False)
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    return {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Roman Urdu / English text",
                      placeholder="Yeh product bohat acha hai..."),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    examples=EXAMPLES,
    title="Roman Urdu Sentiment Classifier",
    description=("DistilBERT fine-tuned on Roman Urdu reviews and social-media comments. "
                 "Type a sentence written in Roman Urdu (code-mixing with English is fine)."),
    article=f"Source code, training and evaluation: [{GITHUB_URL}]({GITHUB_URL})",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
