#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from otl_runner_common import configure_dataset, main


configure_dataset(
    dataset_prefix="twins",
    target_data_dir="twins_stability_streams_repeats50_target",
    good_source_dir="twins_stability_streams_repeats50_source",
    bad_source_dir="twins_bad_stability_streams_repeats50_source",
    output_dir="twins_stability_otl_dr_repeats50_outputs",
)


if __name__ == "__main__":
    main()
