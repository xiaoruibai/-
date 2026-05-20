#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stability_generation_profiles import parse_stability_args, run_stability_generation


def main() -> None:
    args = parse_stability_args("Generate JOBS target/good-source/bad-source stability streams.")
    run_stability_generation(
        dataset_prefix="jobs",
        data_path="jobs_data.csv",
        repeats=args.repeats,
        profile_names=args.profiles,
    )


if __name__ == "__main__":
    main()
