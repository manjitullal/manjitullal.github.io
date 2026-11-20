"""Render notes/*.md into notes/*.html.

Usage:  pip install markdown pymdown-extensions
        python build_notes.py
"""
from pathlib import Path

import markdown

ROOT = Path(__file__).parent / "notes"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · Manjit Ullal</title>
  <link rel="stylesheet" href="../style.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
</head>
<body>
<main class="note">
  <p><a href="../index.html#notes">&larr; back</a></p>
{body}
  <footer><a href="../index.html">Manjit Ullal</a></footer>
</main>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '\\\\[', right: '\\\\]', display: true}},
    {{left: '\\\\(', right: '\\\\)', display: false}}]}})"></script>
</body>
</html>
"""

md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "pymdownx.arithmatex"],
    extension_configs={"pymdownx.arithmatex": {"generic": True}},
)

for src in sorted(ROOT.glob("*.md")):
    text = src.read_text()
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), src.stem)
    body = md.reset().convert(text)
    src.with_suffix(".html").write_text(TEMPLATE.format(title=title, body=body))
    print("built", src.with_suffix(".html").name)
