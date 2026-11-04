# Bitmap font

`font8x8_basic.h` is the original public-domain font source from
[Daniel Hepper's font8x8](https://github.com/dhepper/font8x8/blob/master/font8x8_basic.h),
downloaded for spec 008. Its embedded author/provenance/license notice is retained.

`font8x8.bin` contains the same 128 glyphs as 8 row bytes per glyph, least-significant bit at the left.
Regenerate with `python3 scripts/generate-font.py`. The renderer expands these bits to an R8 atlas at
initialization; no font tool, download, or extra crate is required at build/runtime.
