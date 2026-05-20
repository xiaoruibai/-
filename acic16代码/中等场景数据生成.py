#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moderate_generation_profiles import parse_moderate_args, run_moderate_generation


def main() -> None:
    args = parse_moderate_args("Generate ACIC16 target/good-source/bad-source moderate streams.")
    run_moderate_generation(
        dataset_prefix="acic16",
        data_path="acic16_data.csv",
        repeats=args.repeats,
        profile_names=args.profiles,
    )


if __name__ == "__main__":
    main()
