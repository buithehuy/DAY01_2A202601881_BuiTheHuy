"""
visualize_temperature.py
Gọi call_gemini với temperature [0.0, 0.5, 1.0, 1.5] rồi sinh HTML so sánh.

Chạy:
    python visualize_temperature.py
    python visualize_temperature.py --prompt "Kể một câu chuyện cười"
    python visualize_temperature.py --runs 3   # gọi mỗi temperature 3 lần
"""

import argparse
import sys
import os
import time
import webbrowser

# Đảm bảo import solution từ đúng thư mục
sys.path.insert(0, os.path.dirname(__file__))
from solution.solution import call_gemini

TEMPERATURES = [0.0, 0.5, 1.0, 1.5]
DEFAULT_PROMPT = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."


def collect_responses(prompt: str, runs: int = 1) -> dict:
    """Gọi API và thu thập kết quả."""
    results = {}
    total = len(TEMPERATURES) * runs
    done = 0
    for temp in TEMPERATURES:
        results[temp] = []
        for r in range(runs):
            done += 1
            print(f"  [{done}/{total}] temperature={temp}, lần {r+1}...", end=" ", flush=True)
            t0 = time.time()
            text, latency = call_gemini(prompt, temperature=temp, max_tokens=300)
            print(f"✓ ({latency:.2f}s)")
            results[temp].append({"text": text, "latency": latency})
    return results


# ──────────────────────────────────────────────
#  HTML generation
# ──────────────────────────────────────────────

COLORS = {
    0.0: ("#0ea5e9", "#e0f2fe"),   # sky
    0.5: ("#22c55e", "#dcfce7"),   # green
    1.0: ("#f59e0b", "#fef9c3"),   # amber
    1.5: ("#ef4444", "#fee2e2"),   # red
}

LABELS = {
    0.0: "Rất ổn định",
    0.5: "Cân bằng",
    1.0: "Sáng tạo",
    1.5: "Rất ngẫu nhiên",
}


def build_html(prompt: str, results: dict) -> str:
    cards_html = ""
    for temp in TEMPERATURES:
        accent, bg = COLORS[temp]
        label = LABELS[temp]
        runs = results[temp]
        avg_lat = sum(r["latency"] for r in runs) / len(runs)

        run_blocks = ""
        for i, r in enumerate(runs):
            run_label = f"Lần {i+1}" if len(runs) > 1 else ""
            run_blocks += f"""
            <div class="run-block">
                <div class="run-meta">
                    {f'<span class="run-label">{run_label}</span>' if run_label else ''}
                    <span class="latency">⏱ {r['latency']:.2f}s</span>
                    <span class="char-count">{len(r['text'])} ký tự</span>
                </div>
                <p class="response-text">{r['text'].replace(chr(10), '<br>')}</p>
            </div>
            """

        cards_html += f"""
        <div class="card" style="--accent:{accent}; --bg:{bg};">
            <div class="card-header">
                <div class="temp-badge" style="background:{accent};">T = {temp}</div>
                <div class="card-meta">
                    <span class="label-pill">{label}</span>
                    <span class="avg-lat">avg {avg_lat:.2f}s</span>
                </div>
            </div>
            <div class="runs-container">
                {run_blocks}
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Temperature Comparison — Gemini API</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --surface: #0f172a;
    --surface2: #1e293b;
    --surface3: #334155;
    --text: #f1f5f9;
    --text2: #94a3b8;
    --radius: 16px;
  }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--surface);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem 1rem 4rem;
  }}

  /* ── Header ── */
  .page-header {{
    text-align: center;
    margin-bottom: 2.5rem;
  }}
  .page-header h1 {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .5rem;
  }}
  .page-header .subtitle {{
    color: var(--text2);
    font-size: .95rem;
  }}
  .prompt-box {{
    display: inline-block;
    margin-top: 1rem;
    background: var(--surface2);
    border: 1px solid var(--surface3);
    border-left: 4px solid #38bdf8;
    padding: .75rem 1.25rem;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .85rem;
    color: #e2e8f0;
    max-width: 80ch;
    word-break: break-word;
    text-align: left;
  }}

  /* ── Grid ── */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    max-width: 1600px;
    margin: 0 auto;
  }}

  /* ── Card ── */
  .card {{
    background: var(--surface2);
    border: 1px solid var(--surface3);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform .2s, box-shadow .2s;
  }}
  .card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,.4);
  }}

  .card-header {{
    padding: 1rem 1.25rem .75rem;
    display: flex;
    align-items: center;
    gap: .75rem;
    border-bottom: 1px solid var(--surface3);
  }}

  .temp-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    padding: .3rem .75rem;
    border-radius: 8px;
    letter-spacing: .03em;
    white-space: nowrap;
  }}

  .card-meta {{
    display: flex;
    flex-direction: column;
    gap: .2rem;
  }}
  .label-pill {{
    font-size: .78rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: .06em;
  }}
  .avg-lat {{
    font-size: .75rem;
    color: var(--text2);
  }}

  /* ── Run block ── */
  .runs-container {{
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    flex: 1;
  }}
  .run-block {{
    background: color-mix(in srgb, var(--bg) 15%, var(--surface2));
    border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
    border-radius: 10px;
    padding: .875rem 1rem;
  }}
  .run-meta {{
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: .6rem;
  }}
  .run-label {{
    font-size: .72rem;
    font-weight: 700;
    background: var(--accent);
    color: #fff;
    padding: .15rem .5rem;
    border-radius: 999px;
  }}
  .latency {{
    font-size: .75rem;
    color: var(--text2);
    font-family: 'JetBrains Mono', monospace;
  }}
  .char-count {{
    font-size: .75rem;
    color: var(--text2);
    margin-left: auto;
  }}
  .response-text {{
    font-size: .88rem;
    line-height: 1.65;
    color: #cbd5e1;
    white-space: pre-wrap;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    margin-top: 3rem;
    color: var(--text2);
    font-size: .8rem;
  }}
  .footer code {{
    font-family: 'JetBrains Mono', monospace;
    background: var(--surface2);
    padding: .15rem .4rem;
    border-radius: 4px;
  }}

  /* ── Legend bar ── */
  .legend {{
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: .4rem;
    font-size: .82rem;
    color: var(--text2);
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}
</style>
</head>
<body>

<div class="page-header">
  <h1>🌡️ Temperature Comparison</h1>
  <p class="subtitle">Gemini API — so sánh đầu ra với cùng prompt ở 4 mức temperature</p>
  <div class="prompt-box">💬 "{prompt}"</div>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#0ea5e9"></div>0.0 — Rất ổn định</div>
  <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>0.5 — Cân bằng</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>1.0 — Sáng tạo</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>1.5 — Rất ngẫu nhiên</div>
</div>

<div class="grid">
  {cards_html}
</div>

<div class="footer">
  Sinh bởi <code>visualize_temperature.py</code> | Model: Gemini 2.5 Pro / Flash
</div>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="So sánh temperature trực quan")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt gửi tới API")
    parser.add_argument("--runs", type=int, default=1,
                        help="Số lần gọi mỗi temperature (để thấy tính ngẫu nhiên)")
    parser.add_argument("--out", default="temperature_comparison.html",
                        help="Tên file HTML đầu ra")
    parser.add_argument("--no-open", action="store_true",
                        help="Không tự mở browser sau khi sinh HTML")
    args = parser.parse_args()

    print(f"\n📋 Prompt: {args.prompt!r}")
    print(f"🔁 Số lần/temperature: {args.runs}")
    print(f"🌡️  Temperatures: {TEMPERATURES}\n")
    print("Đang gọi API...")

    results = collect_responses(args.prompt, args.runs)

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    html = build_html(args.prompt, results)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Đã lưu: {out_path}")

    if not args.no_open:
        webbrowser.open(f"file://{out_path}")
        print("🌐 Đã mở browser")


if __name__ == "__main__":
    main()
