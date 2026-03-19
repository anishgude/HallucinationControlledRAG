from __future__ import annotations

import html
import json
import os
import traceback
from typing import List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    os.environ.setdefault("USE_MOCK_LLM", "1")

from src.pipeline import keyword_baseline, naive_gpt_baseline, rag_with_refusal
from src.vector_store import VectorStore


app = FastAPI()


def _ensure_store() -> VectorStore:
    store_dir = "data/vector_store"
    if not os.path.exists(store_dir):
        return VectorStore.build_from_dir("data", store_dir)
    return VectorStore.load(store_dir)


def _format_citations(cited_chunks: List[dict]) -> str:
    if not cited_chunks:
        return "<p><em>No citations returned.</em></p>"
    items = []
    for c in cited_chunks:
        quote = html.escape(c.get("quote", ""))
        doc_id = html.escape(c.get("doc_id", ""))
        offset = html.escape(c.get("offset", ""))
        items.append(f"<li><strong>{doc_id}</strong> ({offset})<br><code>{quote}</code></li>")
    return "<ul>" + "".join(items) + "</ul>"


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)
    return cleaned.strip("_") or "document"


def _render_page(
    question: str = "",
    method: str = "rag",
    answer: dict | None = None,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
    upload_status: str | None = None,
) -> HTMLResponse:
    answer_block = ""
    if answer is not None:
        final_answer = html.escape(answer.get("final_answer", ""))
        refused = "true" if answer.get("refused") else "false"
        citations_html = _format_citations(answer.get("cited_chunks", []))
        answer_block = f"""
        <section class="panel">
          <h2>Answer</h2>
          <p class="answer">{final_answer}</p>
          <p class="meta"><strong>Refused:</strong> {refused}</p>
          <p class="meta"><strong>Latency:</strong> {latency_ms} ms &nbsp; <strong>Cost:</strong> ${cost_usd}</p>
        </section>
        <section class="panel">
          <h2>Citations</h2>
          {citations_html}
        </section>
        <section class="panel">
          <h2>Raw JSON</h2>
          <pre>{html.escape(json.dumps(answer, indent=2))}</pre>
        </section>
        """
    if error:
        answer_block = (
            answer_block
            + f"""
        <section class="panel">
          <h2>Server Error</h2>
          <pre>{html.escape(error)}</pre>
        </section>
        """
        )
    if upload_status:
        answer_block = (
            f"""
        <section class="panel">
          <h2>Upload Status</h2>
          <p class="meta">{html.escape(upload_status)}</p>
        </section>
        """
            + answer_block
        )

    selected = {
        "keyword": "selected" if method == "keyword" else "",
        "naive_gpt": "selected" if method == "naive_gpt" else "",
        "rag": "selected" if method == "rag" else "",
    }

    mode_label = "LIVE MODE" if os.getenv("USE_MOCK_LLM", "0") != "1" else "MOCK MODE"
    key_label = "API KEY SET" if os.getenv("OPENAI_API_KEY") else "API KEY MISSING"
    html_page = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>RAG Hallucination Study Chat</title>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
      <style>
        :root {{
          --bg: #0b0c0f;
          --bg-2: #11151b;
          --ink: #f4f6f8;
          --accent: #4aa6ff;
          --accent-2: #2fd0ff;
          --panel: rgba(255, 255, 255, 0.05);
          --panel-border: rgba(255, 255, 255, 0.12);
          --muted: #9aa6b2;
          --glow: rgba(74, 166, 255, 0.25);
        }}
        body {{
          font-family: "Manrope", system-ui, -apple-system, sans-serif;
          background: var(--bg);
          color: var(--ink);
          margin: 0;
          padding: 32px 20px 56px;
          position: relative;
          overflow-x: hidden;
        }}
        #intro {{
          position: fixed;
          inset: 0;
          background: #000000;
          z-index: 9999;
          display: grid;
          place-items: center;
          transition: opacity 0.8s ease, visibility 0.8s ease;
        }}
        #intro.hidden {{
          opacity: 0;
          visibility: hidden;
        }}
        #intro canvas {{
          width: 100%;
          height: 100%;
          display: block;
        }}
        .intro-title {{
          position: absolute;
          text-align: center;
          font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
          background: linear-gradient(120deg, #7cc6ff, #e8f3ff, #4aa6ff);
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          font-size: clamp(18px, 3vw, 28px);
          letter-spacing: 0.18em;
          text-transform: uppercase;
          opacity: 0.9;
          text-shadow: 0 0 24px rgba(80, 160, 255, 0.45);
          animation: titleGlow 2.6s ease-in-out infinite;
        }}
        body::before {{
          content: "";
          position: fixed;
          inset: -20% 0 0 0;
          background:
            radial-gradient(800px circle at 10% -10%, rgba(244, 185, 66, 0.22), transparent 55%),
            radial-gradient(900px circle at 90% 0%, rgba(242, 143, 59, 0.2), transparent 55%),
            radial-gradient(700px circle at 50% 85%, rgba(61, 122, 140, 0.18), transparent 55%);
          filter: blur(10px);
          z-index: -1;
          animation: drift 16s ease-in-out infinite alternate;
        }}
        .container {{
          max-width: 980px;
          margin: 0 auto;
        }}
        header {{
          display: grid;
          gap: 12px;
          margin-bottom: 22px;
        }}
        h1 {{
          font-size: 38px;
          letter-spacing: -0.03em;
          margin: 0;
        }}
        .subtitle {{
          color: var(--muted);
          margin: 4px 0 0 0;
          font-size: 15px;
          line-height: 1.5;
        }}
        .header-row {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 12px;
        }}
        .badge {{
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          color: #1b1408;
          padding: 6px 12px;
          border-radius: 999px;
          font-size: 12px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          font-weight: 700;
        }}
        form {{
          display: grid;
          gap: 16px;
        }}
        label {{
          display: inline-block;
          margin-bottom: 10px;
          font-size: 15px;
          letter-spacing: 0.01em;
        }}
        textarea {{
          width: 100%;
          min-height: 150px;
          border: 1px solid var(--panel-border);
          border-radius: 14px;
          padding: 16px;
          font-size: 16px;
          resize: vertical;
          background: rgba(9, 12, 16, 0.85);
          color: var(--ink);
          box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
          max-width: 760px;
          margin: 0 auto;
          display: block;
        }}
        select, button {{
          font-size: 16px;
          padding: 12px 14px;
          border-radius: 12px;
        }}
        select {{
          background: rgba(9, 12, 16, 0.85);
          color: var(--ink);
          border: 1px solid var(--panel-border);
        }}
        button {{
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          color: #04111f;
          border: none;
          cursor: pointer;
          font-weight: 700;
          letter-spacing: 0.02em;
          box-shadow: 0 10px 25px var(--glow);
        }}
        button:hover {{
          transform: translateY(-1px);
        }}
        .panel {{
          background: var(--panel);
          padding: 20px;
          border-radius: 18px;
          border: 1px solid var(--panel-border);
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
          margin-top: 16px;
          backdrop-filter: blur(14px);
        }}
        .answer {{
          font-size: 18px;
          line-height: 1.6;
        }}
        .meta {{
          color: var(--muted);
          font-size: 14px;
        }}
        pre {{
          background: #0a0e13;
          color: #d8e4f3;
          padding: 14px;
          border-radius: 12px;
          overflow-x: auto;
          border: 1px solid rgba(255, 255, 255, 0.08);
          font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        }}
        code {{
          background: rgba(255, 255, 255, 0.08);
          padding: 2px 6px;
          border-radius: 6px;
          color: #d8e4f3;
          font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        }}
        .grid {{
          display: grid;
          gap: 16px;
        }}
        .controls {{
          display: grid;
          gap: 12px;
          grid-template-columns: 1fr auto;
          align-items: center;
          margin-top: 6px;
        }}
        .inline {{
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          align-items: center;
        }}
        .pill {{
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid var(--panel-border);
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          color: var(--muted);
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }}
        @keyframes fadeUp {{
          from {{ opacity: 0; transform: translateY(8px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes drift {{
          from {{ transform: translateY(-10px); }}
          to {{ transform: translateY(12px); }}
        }}
        @keyframes glowPulse {{
          0% {{ box-shadow: 0 0 0 rgba(74, 166, 255, 0.0); }}
          50% {{ box-shadow: 0 0 24px rgba(74, 166, 255, 0.25); }}
          100% {{ box-shadow: 0 0 0 rgba(74, 166, 255, 0.0); }}
        }}
        @keyframes titleGlow {{
          0% {{ filter: drop-shadow(0 0 6px rgba(80, 160, 255, 0.2)); letter-spacing: 0.16em; }}
          50% {{ filter: drop-shadow(0 0 18px rgba(80, 160, 255, 0.6)); letter-spacing: 0.2em; }}
          100% {{ filter: drop-shadow(0 0 6px rgba(80, 160, 255, 0.2)); letter-spacing: 0.16em; }}
        }}
        .panel, header {{
          animation: fadeUp 0.45s ease-out;
        }}
        button {{
          transition: transform 0.2s ease, box-shadow 0.3s ease;
        }}
        button:focus-visible {{
          outline: 2px solid var(--accent);
          outline-offset: 2px;
          animation: glowPulse 1.6s ease-in-out infinite;
        }}
        textarea, select {{
          transition: border-color 0.2s ease, box-shadow 0.3s ease;
        }}
        textarea:focus, select:focus {{
          border-color: rgba(74, 166, 255, 0.7);
          box-shadow: 0 0 0 3px rgba(74, 166, 255, 0.15);
          outline: none;
        }}
        .stagger {{
          animation: fadeUp 0.55s ease-out;
        }}
        .ask-wrap {{
          position: relative;
          justify-self: end;
        }}
        .ask-btn {{
          position: relative;
          min-width: 90px;
        }}
        .ask-btn.loading {{
          color: transparent;
          pointer-events: none;
        }}
        .ask-btn.loading::after {{
          content: "";
          position: absolute;
          inset: 0;
          margin: auto;
          width: 18px;
          height: 18px;
          border-radius: 999px;
          border: 2px solid rgba(27, 20, 8, 0.35);
          border-top-color: #1b1408;
          animation: spin 0.7s linear infinite;
        }}
        @keyframes spin {{
          from {{ transform: rotate(0deg); }}
          to {{ transform: rotate(360deg); }}
        }}
        .form-panel {{
          max-width: 860px;
          margin: 0 auto;
        }}
        .file-input {{
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 12px 14px;
          border-radius: 12px;
          border: 1px solid var(--panel-border);
          background: rgba(9, 12, 16, 0.85);
          color: var(--ink);
          cursor: pointer;
          transition: border-color 0.2s ease, box-shadow 0.3s ease;
        }}
        .file-input:hover {{
          border-color: rgba(74, 166, 255, 0.6);
        }}
        .file-input input {{
          display: none;
        }}
        .file-input .chip {{
          padding: 4px 8px;
          border-radius: 999px;
          background: rgba(74, 166, 255, 0.15);
          color: #bfe3ff;
          font-size: 12px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }}
        .file-list {{
          margin-top: 8px;
          color: var(--muted);
          font-size: 13px;
        }}
        @media (max-width: 820px) {{
          .controls {{
            grid-template-columns: 1fr;
          }}
          .ask-wrap {{
            justify-self: start;
          }}
        }}
      </style>
    </head>
    <body>
      <div id="intro">
        <canvas id="swirl"></canvas>
        <div class="intro-title">RAG Hallucination Study</div>
      </div>
      <div class="container">
        <header>
          <div class="header-row">
            <div>
              <h1>RAG Hallucination Study</h1>
              <p class="subtitle">Compare keyword, naive GPT, and RAG + citations + refusal in one place.</p>
            </div>
            <div class="split">
              <span class="badge">local demo</span>
              <span class="badge">{mode_label}</span>
              <span class="badge">{key_label}</span>
            </div>
          </div>
        </header>
        <section class="panel form-panel">
          <form method="post" action="/ask" id="ask-form">
            <div>
              <label for="question"><strong>Your question</strong></label>
              <textarea id="question" name="question" placeholder="Ask about company policies, AetherX manuals, or the heat island study.">{html.escape(question)}</textarea>
            </div>
            <div class="controls">
              <div>
                <span class="pill">Method</span>
                <select id="method" name="method">
                  <option value="keyword" {selected['keyword']}>Keyword baseline</option>
                  <option value="naive_gpt" {selected['naive_gpt']}>Naive GPT baseline</option>
                  <option value="rag" {selected['rag']}>RAG + citations + refusal</option>
                </select>
              </div>
              <div class="ask-wrap">
                <button type="submit" class="stagger ask-btn" id="ask-btn">Ask</button>
              </div>
            </div>
          </form>
        </section>
        <section class="panel form-panel">
          <form method="post" action="/upload" enctype="multipart/form-data">
            <div>
              <label for="docs"><strong>Upload documents</strong></label>
              <label class="file-input" for="docs">
                <span class="chip">Choose files</span>
                <span id="file-label">No files selected</span>
                <input id="docs" name="files" type="file" multiple accept=".md,.txt" />
              </label>
              <div class="file-list" id="file-list"></div>
              <p class="meta">Uploads .md or .txt files and rebuilds the local index.</p>
            </div>
            <div class="controls">
              <div></div>
              <div class="ask-wrap">
                <button type="submit" class="stagger ask-btn">Upload</button>
              </div>
            </div>
          </form>
        </section>
        {answer_block}
      </div>
      <script>
        (function() {{
          const form = document.getElementById("ask-form");
          const button = document.getElementById("ask-btn");
          if (form && button) {{
            form.addEventListener("submit", () => {{
              button.classList.add("loading");
            }});
          }}

          const fileInput = document.getElementById("docs");
          const fileLabel = document.getElementById("file-label");
          const fileList = document.getElementById("file-list");
          if (fileInput && fileLabel && fileList) {{
            fileInput.addEventListener("change", () => {{
              const names = Array.from(fileInput.files || []).map((f) => f.name);
              fileLabel.textContent = names.length ? (names.length + " file(s) selected") : "No files selected";
              fileList.textContent = names.join(", ");
            }});
          }}

          const intro = document.getElementById("intro");
          const canvas = document.getElementById("swirl");
          const ctx = canvas.getContext("2d");
          let w = 0, h = 0, particles = [], t = 0;

          function resize() {{
            w = canvas.width = window.innerWidth;
            h = canvas.height = window.innerHeight;
          }}
          function spawn() {{
            const count = 1800;
            const ringCount = 420;
            const swirlCount = count - ringCount;
            particles = [];
            for (let i = 0; i < swirlCount; i++) {{
              const radius = 80 + Math.random() * Math.max(w, h) * 0.55;
              particles.push({{
                x: w * 0.5,
                y: h * 0.5,
                prevX: w * 0.5,
                prevY: h * 0.5,
                r: 0.9 + Math.random() * 2.4,
                s: 0.6 + Math.random() * 2.0,
                a: Math.random() * Math.PI * 2,
                hue: 200 + Math.random() * 40,
                radius: radius,
                drift: 0.6 + Math.random() * 1.2,
                arm: Math.floor(Math.random() * 3),
                shape: Math.floor(Math.random() * 3),
                z: Math.random(),
                phaseOffset: Math.random() * Math.PI * 2,
                phaseSpeed: 0.001 + Math.random() * 0.003,
                xDrift: 2 + Math.random() * 6,
                trail: Math.random() < 0.45,
                kind: "swirl",
              }});
            }}
            for (let i = 0; i < ringCount; i++) {{
              const radius = Math.max(w, h) * 0.32 + Math.random() * 34;
              particles.push({{
                x: w * 0.5,
                y: h * 0.5,
                prevX: w * 0.5,
                prevY: h * 0.5,
                r: 1.0 + Math.random() * 2.6,
                s: 0.8 + Math.random() * 1.6,
                a: Math.random() * Math.PI * 2,
                hue: 205 + Math.random() * 50,
                radius: radius,
                drift: 0.6 + Math.random() * 1.2,
                arm: Math.floor(Math.random() * 3),
                shape: Math.floor(Math.random() * 3),
                z: Math.random(),
                phaseOffset: Math.random() * Math.PI * 2,
                phaseSpeed: 0.001 + Math.random() * 0.003,
                xDrift: 2 + Math.random() * 6,
                trail: Math.random() < 0.45,
                kind: "ring",
              }});
            }}
          }}
          function draw() {{
            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, w, h);
            ctx.save();
            ctx.translate(w * 0.5, h * 0.5);
            ctx.rotate(t * 0.001);
            ctx.translate(-w * 0.5, -h * 0.5);

            particles.forEach((p, i) => {{
              const armOffset = (Math.PI * 2 / 3) * p.arm;
              const basePhase = t * 0.01 + i * 0.04 + p.phaseOffset + t * p.phaseSpeed;
              const swirl = Math.sin(t * 0.006 + i * 0.02) * 0.9;
              if (p.kind === "ring") {{
                const phase = t * 0.02 + p.a + p.phaseOffset * 0.5;
                const wobble = Math.sin(t * 0.035 + i) * 14 + Math.cos(t * 0.025 + i * 0.4) * 9;
                const radius = p.radius + wobble;
                const x = Math.cos(phase) * radius + Math.sin(t * 0.004 + p.phaseOffset) * p.xDrift;
                const y = Math.sin(phase) * radius * (0.6 + Math.sin(t * 0.012 + i) * 0.12);
                p.x = w * 0.5 + x;
                p.y = h * 0.5 + y;
              }} else {{
                const radius = p.radius + Math.sin(t * 0.02 + i) * 10;
                const x = Math.cos(basePhase + armOffset + swirl) * radius + Math.sin(t * 0.004 + p.phaseOffset) * p.xDrift;
                const y = Math.sin(basePhase + armOffset - swirl) * radius * 0.7;
                p.x = w * 0.5 + x;
                p.y = h * 0.5 + y;
              }}

              if (p.trail) {{
                ctx.beginPath();
                ctx.strokeStyle = "rgba(90, 190, 255, 0.55)";
                ctx.lineWidth = 1.8;
                ctx.shadowColor = "rgba(90, 190, 255, 0.45)";
                ctx.shadowBlur = 10;
                ctx.moveTo(p.prevX, p.prevY);
                ctx.lineTo(p.x, p.y);
                ctx.stroke();
              }}

              ctx.beginPath();
              const depth = 0.4 + p.z * 1.1;
              const size = p.r * depth;
              const alpha = 0.25 + p.z * 0.75;
              ctx.fillStyle = "hsla(" + p.hue + ", 90%, 68%, " + alpha.toFixed(2) + ")";
              ctx.shadowColor = "rgba(70, 170, 255, " + (0.2 + p.z * 0.6).toFixed(2) + ")";
              ctx.shadowBlur = 8 + p.z * 18;
              if (p.shape === 0) {{
                ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
                ctx.fill();
              }} else if (p.shape === 1) {{
                const side = size * 2.2;
                ctx.fillRect(p.x - side / 2, p.y - side / 2, side, side);
              }} else {{
                const side = size * 2.4;
                ctx.moveTo(p.x, p.y - side / 1.6);
                ctx.lineTo(p.x - side / 1.6, p.y + side / 1.6);
                ctx.lineTo(p.x + side / 1.6, p.y + side / 1.6);
                ctx.closePath();
                ctx.fill();
              }}
              p.prevX = p.x;
              p.prevY = p.y;
            }});
            ctx.shadowBlur = 0;
            ctx.restore();
            t += 1;
            requestAnimationFrame(draw);
          }}

          resize();
          spawn();
          draw();
          window.addEventListener("resize", resize);

          setTimeout(() => {{
            intro.classList.add("hidden");
          }}, 2200);
        }})();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html_page)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _render_page()


@app.post("/ask", response_class=HTMLResponse)
def ask(question: str = Form(...), method: str = Form("rag")) -> HTMLResponse:
    store = _ensure_store()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        if method == "keyword":
            result, usage, latency = keyword_baseline(question, store, model)
        elif method == "naive_gpt":
            result, usage, latency = naive_gpt_baseline(question, model)
        else:
            result, usage, latency, _ = rag_with_refusal(question, store, model)

        cost = 0.0
        if usage:
            from src.llm import estimate_cost

            cost = round(estimate_cost(usage, model), 6)

        return _render_page(
            question=question,
            method=method,
            answer=result,
            latency_ms=round(latency, 2),
            cost_usd=cost,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        return _render_page(question=question, method=method, error=error)


@app.post("/upload", response_class=HTMLResponse)
def upload(files: List[UploadFile] = File(...)) -> HTMLResponse:
    saved = 0
    skipped = 0
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for file in files:
        name = _safe_filename(file.filename or "")
        base, ext = os.path.splitext(name)
        ext = ext.lower()
        if ext not in (".md", ".txt"):
            skipped += 1
            continue
        if ext == ".txt":
            ext = ".md"
        dest_name = f"{base}{ext}"
        dest_path = os.path.join(data_dir, dest_name)
        content = file.file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        saved += 1

    VectorStore.build_from_dir(data_dir, os.path.join(data_dir, "vector_store"))
    status = f"Uploaded {saved} file(s). Skipped {skipped} unsupported file(s). Index rebuilt."
    return _render_page(upload_status=status)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)
