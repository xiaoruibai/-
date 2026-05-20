#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from moderate_generation_profiles import parse_moderate_args, run_moderate_generation


def main() -> None:
    args = parse_moderate_args("Generate Twins target/good-source/bad-source moderate streams.")
    run_moderate_generation(
        dataset_prefix="twins",
        data_path="twin_pairs_X_3years_samesex.csv",
        repeats=args.repeats,
        profile_names=args.profiles,
    )


if __name__ == "__main__":
    main()
