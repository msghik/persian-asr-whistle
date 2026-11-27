#!/usr/bin/env python3
"""REPORT.md -> REPORT.pdf (A4, 2 cm margins, 10 pt body; Noto fonts for Persian). Fails if not exactly 1 page.

  python scripts/report_pdf.py [REPORT.md] [REPORT.pdf]
Needs: pip weasyprint==62.3 pydyf==0.10.0 markdown==3.7; apt libpango-1.0-0 fonts-noto-core poppler-utils.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown
import weasyprint

REPO = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: "Noto Serif", "Noto Naskh Arabic", "DejaVu Serif", serif; font-size: 10pt;
       line-height: 1.2; color: #111; }
h1 { font-size: 13pt; margin: 0 0 1pt 0; line-height: 1.2; }
h1 + p { margin-top: 0; color: #333; }
p { margin: 3pt 0; }
ul { margin: 2pt 0 2pt 0; padding-left: 13pt; }
li { margin: 0.5pt 0; }
table { border-collapse: collapse; margin: 2pt 0 2pt 0; font-size: 10pt; }
th, td { border-bottom: 0.5pt solid #999; padding: 0.6pt 6pt; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { border-bottom: 0.8pt solid #333; font-weight: 600; }
bdi { font-family: "Noto Naskh Arabic", "Noto Sans Arabic", sans-serif; }
"""


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "REPORT.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "REPORT.pdf"
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables"])
    html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    weasyprint.HTML(string=html).write_pdf(out)
    info = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True).stdout
    pages = int(next(l.split()[-1] for l in info.splitlines() if l.startswith("Pages:")))
    print(f"{out}: {pages} page(s)")
    sys.exit(0 if pages == 1 else 1)


if __name__ == "__main__":
    main()
