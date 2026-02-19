"""Small web service to classify tweets using a Hugging Face model.

FastAPI app with a minimal HTML UI and a JSON API.
It tries to load a local model in ./models/sentiment and falls back
to the hub model `distilbert-base-uncased-finetuned-sst-2-english`.

Run (development):
    pip install fastapi uvicorn transformers torch
    uvicorn tareas.20260216_LLM_web_service:app --reload --port 8000

Open http://127.0.0.1:8000
"""

from pathlib import Path
import os
from typing import List

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from transformers import pipeline
except Exception:
    raise RuntimeError("Please install transformers and torch: pip install transformers torch")

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/sentiment")
HUB_MODEL = os.environ.get("HUB_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")


def load_pipeline(model_dir: str = MODEL_DIR, hub_model: str = HUB_MODEL):
    path = Path(model_dir)
    if path.exists():
        try:
            nlp = pipeline("sentiment-analysis", model=str(path))
            print(f"Loaded local model from: {path}")
            return nlp
        except Exception as e:
            print(f"Failed to load local model from {path}: {e}")

    print(f"Loading model from Hub: {hub_model}")
    nlp = pipeline("sentiment-analysis", model=hub_model)
    return nlp


app = FastAPI()

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Tweet Sentiment</title>
    <style>body{font-family:Arial; margin:40px;} textarea{width:100%;height:120px;} .result{margin-top:20px;padding:10px;border-radius:6px;background:#f4f4f4}</style>
  </head>
  <body>
    <h2>Tweet Sentiment Classifier</h2>
    <form method="post" action="/predict">
      <label for="tweet">Paste a tweet (or several, one per line):</label><br>
      <textarea name="tweet" id="tweet">{example}</textarea><br>
      <button type="submit">Classify</button>
    </form>
    {results_block}
  </body>
</html>
"""


@app.on_event("startup")
def startup_event():
    # Load pipeline once
    print('Starting app, loading model pipeline...')
    app.state.nlp = load_pipeline()
    print('Pipeline loaded (or attempted).')


def render_results(example: str, results: List[dict]) -> str:
    # Use simple replace instead of str.format because the HTML contains
    # raw braces in CSS which would break format() parsing.
    if not results:
        return HTML.replace("{example}", example).replace("{results_block}", "")
    items = "\n".join([f"<li><strong>{r['label']}</strong> (score={r['score']:.3f}) — {r['text']}</li>" for r in results])
    block = f"<div class=\"result\"><h3>Results</h3><ul>{items}</ul></div>"
    return HTML.replace("{example}", example).replace("{results_block}", block)


@app.get("/", response_class=HTMLResponse)
def index():
    example = "I love this new phone! It's awesome."
    print('Serving index page')
    return render_results(example, [])


@app.get('/health')
def health():
    # Simple health check
    return {"status": "ok"}


@app.get('/raw', response_class=HTMLResponse)
def raw_test():
    # Very small HTML to test server rendering independent of pipeline
    return HTMLResponse('<html><body><h1>raw test OK</h1></body></html>')


@app.post("/predict", response_class=HTMLResponse)
async def predict(tweet: str = Form(...)):
    text = tweet.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        lines = [text]

    nlp = app.state.nlp
    outputs = nlp(lines, truncation=True)
    results = []
    for txt, out in zip(lines, outputs):
        results.append({"label": out.get("label"), "score": float(out.get("score", 0.0)), "text": txt})

    return render_results(text, results)


@app.post("/api/predict")
async def api_predict(payload: dict):
    # expected payload: {"texts": ["...", ...]}
    texts = payload.get("texts") if isinstance(payload, dict) else None
    if not texts or not isinstance(texts, list):
        return JSONResponse({"error": "send JSON like {\"texts\": [\"...\"]}"}, status_code=400)

    nlp = app.state.nlp
    outputs = nlp(texts, truncation=True)
    results = []
    for txt, out in zip(texts, outputs):
        results.append({"text": txt, "label": out.get("label"), "score": float(out.get("score", 0.0))})
    return {"results": results}


if __name__ == '__main__':
    # fallback run for convenience; prefer: uvicorn tareas.20260216_LLM_web_service:app --reload --port 8000
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
