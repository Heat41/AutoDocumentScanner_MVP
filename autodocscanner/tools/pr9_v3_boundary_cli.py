"""Locally inspect V3 Stable/Adaptive boundary evidence without writing crops.

Usage: python -m autodocscanner.tools.pr9_v3_boundary_cli /path/to/photo.jpg
Only aggregate V3 diagnostic metrics are printed, never image pixels or paths.
"""
from __future__ import annotations

import argparse
import json

import cv2

from autodocscanner.core.pr9_boundary_audit import diagnose_v3_image_pr9


def main(argv=None, *, evaluator=None):
    parser = argparse.ArgumentParser(description="PR9 V3-only boundary audit (no V2, no auto-crop)")
    parser.add_argument("image", help="Local source photo (stays on this computer)")
    args = parser.parse_args(argv)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        parser.error("Foto tidak bisa dibaca")
    result = (diagnose_v3_image_pr9 if evaluator is None else evaluator)(image)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
