#!/usr/bin/env python3
"""Check that the published paper PDFs were built the way ICML expects.

Requires poppler-utils (pdfinfo, pdffonts, pdftotext).  The checks catch the
failure modes of non-pdfTeX builds: substituted non-Times body fonts,
non-Type-1 font programs, and the icml2026.sty fallback running head.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ("sft_paper", "rl_paper", "eval_paper")
SUPPRESSED_HEAD = "Title Suppressed Due to Excessive Size"
BODY_FONT_MARKERS = ("NimbusRomNo9L", "Times", "utmr8a")


def run(tool: str, *args: str) -> str:
    return subprocess.run(
        [tool, *args], check=True, capture_output=True, text=True
    ).stdout


def font_rows(pdf: Path) -> list[tuple[str, str]]:
    rows = []
    for line in run("pdffonts", str(pdf)).splitlines()[2:]:
        parts = line.split()
        if not parts:
            continue
        # Columns: name, type (one or more words), encoding, emb, sub, uni, id.
        name = parts[0]
        font_type = " ".join(parts[1:-6])
        rows.append((name, font_type))
    return rows


def check_pdf(pdf: Path) -> list[str]:
    if not pdf.is_file():
        return [f"{pdf}: missing"]
    errors = []
    info = run("pdfinfo", str(pdf))
    producer = next(
        (l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("Producer:")),
        "",
    )
    if not producer.startswith("pdfTeX"):
        errors.append(f"{pdf}: producer is {producer!r}, expected pdfTeX")
    if "Page size:       612 x 792 pts" not in info:
        errors.append(f"{pdf}: page size is not US letter")

    fonts = font_rows(pdf)
    bad_types = sorted({t for _, t in fonts if t != "Type 1"})
    if bad_types:
        errors.append(f"{pdf}: non-Type-1 fonts present: {', '.join(bad_types)}")
    if not any(m in name for name, _ in fonts for m in BODY_FONT_MARKERS):
        errors.append(f"{pdf}: no Times body font found")

    if SUPPRESSED_HEAD in run("pdftotext", str(pdf), "-"):
        errors.append(f"{pdf}: running head fell back to {SUPPRESSED_HEAD!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path, help="defaults to the three papers")
    args = parser.parse_args()

    missing = [t for t in ("pdfinfo", "pdffonts", "pdftotext") if shutil.which(t) is None]
    if missing:
        print(f"error: missing poppler-utils tools: {', '.join(missing)}", file=sys.stderr)
        return 127

    pdfs = args.pdfs or [ROOT / p / "main.pdf" for p in PAPERS]
    errors = [e for pdf in pdfs for e in check_pdf(pdf)]
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    for pdf in pdfs:
        print(f"ok: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
