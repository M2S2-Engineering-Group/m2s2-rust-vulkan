#!/usr/bin/env python3
"""Regenerate the bundled row-bit font data from its public-domain source header."""
from pathlib import Path
import re

assets = Path(__file__).resolve().parent.parent / "assets"
rows = re.findall(r"\{((?:\s*0x[0-9A-Fa-f]+,?){8})\s*\}", (assets / "font8x8_basic.h").read_text())
assert len(rows) == 128, "Expected 128 eight-row glyphs"
(assets / "font8x8.bin").write_bytes(bytes(
    int(value, 16) for row in rows for value in re.findall(r"0x[0-9A-Fa-f]+", row)
))
