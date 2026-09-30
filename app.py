"""Receipt extractor: PaddleOCR -> GPT-4o -> {sender_name, amount, date}."""
import json
import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI
from paddleocr import PaddleOCR

load_dotenv()

app = Flask(__name__, static_folder="static")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Loaded once at startup (first run downloads the models).
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    # oneDNN crashes under Paddle 3.x's PIR executor on Windows CPU
    # ("ConvertPirAttribute2RuntimeAttribute not support").
    enable_mkldnn=False,
)

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

SYSTEM_PROMPT = """You extract fields from OCR text of a receipt or payment/transfer slip.
The OCR text may contain errors, split lines, or noise. Return ONLY a JSON object:
{
  "sender_name": string or null,   // person or business that sent/issued the payment
  "amount": string or null,        // total amount paid, with currency if visible (e.g. "125.50 USD")
  "date": string or null           // transaction date, normalized to YYYY-MM-DD when possible
}
Use null for anything you cannot find. Do not invent values."""


def run_ocr(path: str) -> str:
    lines = []
    for res in ocr.predict(path):
        lines.extend(res["rec_texts"])
    return "\n".join(lines)


def extract_fields(text: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR text:\n\n{text}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/extract")
def extract():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="No file uploaded."), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type: {ext}"), 400

    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        file.save(path)
        text = run_ocr(path)
    finally:
        os.remove(path)

    if not text.strip():
        return jsonify(error="No text detected in the image.", ocr_text=""), 422

    try:
        fields = extract_fields(text)
    except Exception as e:  # surface API errors to the page
        return jsonify(error=f"GPT extraction failed: {e}", ocr_text=text), 502

    return jsonify(ocr_text=text, fields=fields)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
