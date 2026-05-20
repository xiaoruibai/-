#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream_generation_common import StreamGenerationConfig, run_generation


def sample_beta(rng: np.random.Generator, d: int) -> np.ndarray:
    vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    probs = np.array([0.35, 0.15, 0.1, 0.2, 0.2], dtype=float)
    beta = rng.choice(vals, size=d, p=probs).astype(float)
    if np.linalg.norm(beta) < 1e-12:
        beta[rng.integers(0, d)] = 0.1
    return beta


def mu_nonlinear(X: np.ndarray, beta3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nb = np.linalg.norm(beta3) + 1e-12
    mu_m1 = np.exp((X + 0.05) @ beta3) / np.exp(nb)
    mu_p1 = (X @ beta3 + 0.05) / nb
    return mu_m1, mu_p1


def main() -> None:
    run_generation(
        StreamGenerationConfig(
            dataset_prefix="acic16",
            data_path="acic16_data.csv",
            outdir="acic16_ethos_streams_repeats50_source",
            t=3000,
            repeats=50,
            eps=0.05,
            noise_var=0.1,
            base_seed=2026,
            prop_subsample=3000,
            env_prefix="acic16_good",
            beta_sampler=sample_beta,
            response_surface=mu_nonlinear,
            propensity_gamma_scale=1.5,
            done_message="All acic16 good/source nonlinear-only repeat streams generated successfully.",
        )
    )


if __name__ == "__main__":
    main()
