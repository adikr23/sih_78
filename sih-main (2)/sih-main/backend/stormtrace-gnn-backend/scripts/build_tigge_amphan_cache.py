"""Precompute the real NCMRWF TIGGE Amphan geographic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from src.tigge_amphan import build_tigge_amphan_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", default=settings.neps_g_forecast_path)
    parser.add_argument("--cache-dir", default=settings.imdaa_processed_dir)
    parser.add_argument("--training-epochs", type=int, default=20)
    args = parser.parse_args()
    if args.training_epochs < 1:
        parser.error("--training-epochs must be at least one")
    result = build_tigge_amphan_cache(
        Path(args.cache_dir),
        forecast_path=args.forecast,
        training_epochs=args.training_epochs,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
