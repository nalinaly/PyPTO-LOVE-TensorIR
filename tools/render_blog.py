#!/usr/bin/env python3
"""Render a Markdown report as one self-contained, offline HTML file."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SOURCE = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)


STYLE = """
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #5f6b76;
  --line: #d9e0e6;
  --paper: #ffffff;
  --soft: #f5f7fa;
  --accent: #2457a7;
  --note: #fff7d6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--soft);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
    "Microsoft YaHei", sans-serif;
  font-size: 17px;
  line-height: 1.72;
}
main {
  width: min(1040px, calc(100% - 32px));
  margin: 32px auto;
  padding: 48px clamp(24px, 6vw, 76px) 72px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 12px 36px rgba(30, 45, 65, 0.08);
}
h1 { margin-top: 2.1em; border-bottom: 1px solid var(--line); padding-bottom: .35em; }
h1:first-child { margin-top: 0; font-size: 2.15em; }
h2 { margin-top: 1.7em; }
h3 { margin-top: 1.4em; }
a { color: var(--accent); text-underline-offset: 3px; }
code {
  font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: .9em;
  background: #eef2f7;
  border-radius: 4px;
  padding: .12em .3em;
}
pre {
  overflow-x: auto;
  padding: 18px 20px;
  background: #111827;
  color: #eef2ff;
  border-radius: 8px;
  line-height: 1.55;
}
pre code { padding: 0; background: transparent; color: inherit; }
blockquote {
  margin: 1.25em 0;
  padding: 12px 18px;
  border-left: 4px solid #d69e2e;
  background: var(--note);
}
table { width: 100%; border-collapse: collapse; display: block; overflow-x: auto; }
th, td { padding: 9px 12px; border: 1px solid var(--line); text-align: left; }
th { background: #edf3fb; }
img { display: block; max-width: 100%; height: auto; margin: 22px auto; }
hr { border: 0; border-top: 1px solid var(--line); }
.render-meta { margin-top: 48px; color: var(--muted); font-size: .84em; }
@media print {
  body { background: #fff; }
  main { width: 100%; margin: 0; border: 0; box-shadow: none; }
}
""".strip()


def _image_data_uri(source: str, markdown_path: Path) -> str:
    if source.startswith("data:"):
        return source
    if source.startswith(("http://", "https://", "//")):
        raise ValueError(f"remote image is not allowed in single-file HTML: {source}")
    image_path = (markdown_path.parent / source).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Markdown image does not exist: {source}")
    mime, _encoding = mimetypes.guess_type(image_path.name)
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"unsupported image type: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render(markdown_path: Path) -> str:
    try:
        import markdown
    except ImportError as error:  # pragma: no cover - exercised by environment gate
        raise RuntimeError("Python-Markdown is required to render the blog") from error

    markdown_path = markdown_path.resolve()
    text = markdown_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=("fenced_code", "tables", "toc", "sane_lists"),
        output_format="html5",
    )

    def replace_image(match: re.Match[str]) -> str:
        return (
            match.group(1)
            + html.escape(_image_data_uri(match.group(2), markdown_path), quote=True)
            + match.group(3)
        )

    body = IMAGE_SOURCE.sub(replace_image, body)
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else markdown_path.stem
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>",
            f"<style>{STYLE}</style>",
            "</head>",
            "<body>",
            f"<main>{body}<p class=\"render-meta\">Generated from "
            f"{html.escape(markdown_path.name)}; all visual assets are embedded.</p></main>",
            "</body>",
            "</html>",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
