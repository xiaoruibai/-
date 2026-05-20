#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from stability_generation_profiles import parse_stability_args, run_stability_generation


def main() -> None:
    args = parse_stability_args("Generate Twins target/good-source/bad-source stability streams.")
    run_stability_generation(
        dataset_prefix="twins",
        data_path="twin_pairs_X_3years_samesex.csv",
        repeats=args.repeats,
        profile_names=args.profiles,
    )


if __name__ == "__main__":
    main()
