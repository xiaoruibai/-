#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import os
import re
import sys
from types import ModuleType


_MODULE_NAME = "_ihdp_double_line_otl"
_MODULE_PATH = os.path.join(os.path.dirname(__file__), "双线OTL.py")


def _load_runner() -> ModuleType:
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OTL runner from {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_runner = _load_runner()
_configured_target_data_dir: str | None = None


def _user_supplied_n_repeats() -> bool:
    return any(arg == "--n-repeats" or arg.startswith("--n-repeats=") for arg in sys.argv[1:])


def _detect_available_repeats(data_dir: str) -> int | None:
    if not os.path.isdir(data_dir):
        return None
    repeats_by_env: dict[str, set[int]] = {}
    for fname in os.listdir(data_dir):
        match = _runner.REPEAT_ENV_FILE_REGEX.match(fname)
        if match is None:
            continue
        env_name = str(match.group(1))
        repeat_id = int(match.group(5))
        repeats_by_env.setdefault(env_name, set()).add(repeat_id)
    if not repeats_by_env:
        return None
    counts = {env_name: len(repeats) for env_name, repeats in repeats_by_env.items()}
    detected = min(counts.values())
    expected = {
        env_name: set(range(1, detected + 1))
        for env_name in repeats_by_env
    }
    if any(repeats_by_env[env_name] != expected[env_name] for env_name in repeats_by_env):
        detail = {
            env_name: sorted(repeats)
            for env_name, repeats in sorted(repeats_by_env.items())
        }
        raise ValueError(
            "Detected non-contiguous repeat inventory in target data directory. "
            f"Expected every environment to contain repeat01..repeat{detected:02d}. "
            f"Observed: {detail}"
        )
    return detected


def configure_dataset(
    dataset_prefix: str,
    target_data_dir: str,
    good_source_dir: str,
    bad_source_dir: str,
    output_dir: str,
) -> None:
    """Configure the shared double-line OTL runner for one IHDP scenario."""
    global _configured_target_data_dir
    _configured_target_data_dir = target_data_dir
    _runner.DEFAULT_TARGET_DATA_DIR = target_data_dir
    _runner.DEFAULT_GOOD_SOURCE_DIR = good_source_dir
    _runner.DEFAULT_BAD_SOURCE_DIR = bad_source_dir
    _runner.DEFAULT_OUTPUT_DIR = output_dir
    _runner.REPEAT_ENV_FILE_REGEX = re.compile(
        rf"^({re.escape(dataset_prefix)}_(linear|nonlinear)_(switching|linear)_K(\d+))_repeat(\d+)\.csv$"
    )
    _runner.LEGACY_ENV_FILE_REGEX = re.compile(
        rf"^({re.escape(dataset_prefix)}_(linear|nonlinear)_(switching|linear)_K(\d+)\.csv)$"
    )


def main() -> None:
    if not _user_supplied_n_repeats() and _configured_target_data_dir is not None:
        detected_repeats = _detect_available_repeats(_configured_target_data_dir)
        if detected_repeats is not None:
            _runner.N_REPEATS = detected_repeats
            print(
                f"[INFO] Auto-detected n_repeats={detected_repeats} "
                f"from {_configured_target_data_dir}. "
                "Pass --n-repeats explicitly to override."
            )
    _runner.mp.freeze_support()
    _runner.main()
