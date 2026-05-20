#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from otl_runner_two_layer_common import configure_dataset, main


configure_dataset(
    dataset_prefix="acic16",
    target_data_dir="acic16_moderate_streams_repeats50_target",
    good_source_dir="acic16_moderate_streams_repeats50_source",
    bad_source_dir="acic16_bad_moderate_streams_repeats50_source",
    output_dir="acic16_moderate_two_layer_otl_dr_repeats50_outputs",
)


if __name__ == "__main__":
    main()
