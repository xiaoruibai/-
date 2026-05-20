#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from moderate_generation_profiles import parse_moderate_args, run_moderate_generation


def main() -> None:
    args = parse_moderate_args("Generate IHDP target/good-source/bad-source moderate streams.")
    run_moderate_generation(
        dataset_prefix="ihdp",
        data_path="ihdp_data_1.csv",
        repeats=args.repeats,
        profile_names=args.profiles,
    )


if __name__ == "__main__":
    main()
