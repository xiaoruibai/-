#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from otl_runner_common import configure_dataset, main


configure_dataset(
    dataset_prefix="jobs",
    target_data_dir="jobs_ethos_streams_repeats50_target",
    good_source_dir="jobs_ethos_streams_repeats50_source",
    bad_source_dir="jobs_bad_ethos_streams_repeats50_source",
    output_dir="jobs_otl_dr_repeats50_outputs",
)


if __name__ == "__main__":
    main()
