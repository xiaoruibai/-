#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Iterable

import numpy as np

from stream_generation_common import StreamGenerationConfig, run_generation


@dataclass(frozen=True)
class StabilityProfile:
    name: str
    env_prefix_suffix: str
    outdir_suffix: str
    t: int
    base_seed: int
    propensity_gamma_scale: float
    beta_values: tuple[float, ...]
    beta_probs: tuple[float, ...]
    quad_scale: float
    inter_scale: float
    beta_jitter: float
    base: float
    tau_base: float
    a0: float
    a1: float
    b0: float
    b1: float
    c0: float
    c1: float
    offset0: float
    offset1: float
    phase: float
    sign_flip_rate: float = 0.0


TARGET_PROFILE = StabilityProfile(
    name="target",
    env_prefix_suffix="",
    outdir_suffix="stability_streams_repeats50_target",
    t=1000,
    base_seed=2026,
    propensity_gamma_scale=1.0,
    beta_values=(0.0, 0.08, 0.16, 0.24, 0.32),
    beta_probs=(0.35, 0.20, 0.20, 0.15, 0.10),
    quad_scale=0.55,
    inter_scale=0.35,
    beta_jitter=0.015,
    base=0.10,
    tau_base=0.22,
    a0=0.55,
    a1=0.38,
    b0=0.26,
    b1=0.18,
    c0=0.12,
    c1=0.08,
    offset0=0.02,
    offset1=0.06,
    phase=0.15,
)

GOOD_SOURCE_PROFILE = StabilityProfile(
    name="good_source",
    env_prefix_suffix="good",
    outdir_suffix="stability_streams_repeats50_source",
    t=3000,
    base_seed=2027,
    propensity_gamma_scale=1.15,
    beta_values=(0.0, 0.075, 0.15, 0.23, 0.30),
    beta_probs=(0.34, 0.21, 0.20, 0.15, 0.10),
    quad_scale=0.50,
    inter_scale=0.32,
    beta_jitter=0.02,
    base=0.12,
    tau_base=0.20,
    a0=0.52,
    a1=0.35,
    b0=0.23,
    b1=0.16,
    c0=0.11,
    c1=0.075,
    offset0=0.04,
    offset1=0.08,
    phase=0.22,
)

BAD_SOURCE_PROFILE = StabilityProfile(
    name="bad_source",
    env_prefix_suffix="bad",
    outdir_suffix="bad_stability_streams_repeats50_source",
    t=3000,
    base_seed=520,
    propensity_gamma_scale=1.65,
    beta_values=(0.0, -0.10, -0.18, -0.28, 0.14),
    beta_probs=(0.20, 0.25, 0.25, 0.20, 0.10),
    quad_scale=-0.80,
    inter_scale=0.75,
    beta_jitter=0.04,
    base=-0.08,
    tau_base=-0.12,
    a0=-0.62,
    a1=-0.42,
    b0=0.42,
    b1=-0.30,
    c0=0.24,
    c1=-0.20,
    offset0=-0.22,
    offset1=0.28,
    phase=1.10,
    sign_flip_rate=0.45,
)

PROFILES = (TARGET_PROFILE, GOOD_SOURCE_PROFILE, BAD_SOURCE_PROFILE)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v
    return v / norm


def make_beta_sampler(profile: StabilityProfile):
    def sample_beta(rng: np.random.Generator, d: int) -> np.ndarray:
        vals = np.asarray(profile.beta_values, dtype=float)
        probs = np.asarray(profile.beta_probs, dtype=float)
        beta = rng.choice(vals, size=d, p=probs).astype(float)
        if profile.beta_jitter > 0:
            beta = beta + rng.normal(0.0, profile.beta_jitter, size=d)
        if profile.sign_flip_rate > 0:
            flips = rng.random(d) < profile.sign_flip_rate
            beta[flips] = -beta[flips]
        if np.linalg.norm(beta) < 1e-12:
            beta[rng.integers(0, d)] = vals[1] if len(vals) > 1 else 0.1
        beta_quad = profile.quad_scale * _normalize(beta + rng.normal(0.0, 0.03, size=d))
        beta_inter = profile.inter_scale * _normalize(rng.normal(0.0, 1.0, size=d))
        return np.concatenate([beta, beta_quad, beta_inter]).astype(float)

    return sample_beta


def make_response_surface(profile: StabilityProfile):
    def response_surface(X: np.ndarray, packed_beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = X.shape[1]
        beta = packed_beta[:d]
        beta_quad = packed_beta[d: 2 * d]
        beta_inter = packed_beta[2 * d: 3 * d]
        score = X @ beta
        quad = (X ** 2) @ beta_quad
        inter = np.sin(X @ beta_inter + profile.phase)
        mu_m1 = (
            profile.base
            + profile.a0 * np.tanh(score + profile.offset0)
            + profile.b0 * quad
            + profile.c0 * inter
        )
        mu_p1 = (
            mu_m1
            + profile.tau_base
            + profile.a1 * np.tanh(score + profile.offset1)
            + profile.b1 * quad
            + profile.c1 * inter
        )
        return mu_m1, mu_p1

    return response_surface


def env_prefix(dataset_prefix: str, profile: StabilityProfile) -> str:
    if not profile.env_prefix_suffix:
        return dataset_prefix
    return f"{dataset_prefix}_{profile.env_prefix_suffix}"


def outdir(dataset_prefix: str, profile: StabilityProfile) -> str:
    return f"{dataset_prefix}_{profile.outdir_suffix}"


def build_config(
    dataset_prefix: str,
    data_path: str,
    profile: StabilityProfile,
    repeats: int,
) -> StreamGenerationConfig:
    return StreamGenerationConfig(
        dataset_prefix=dataset_prefix,
        data_path=data_path,
        outdir=outdir(dataset_prefix, profile),
        t=profile.t,
        repeats=repeats,
        eps=0.05,
        noise_var=0.05,
        base_seed=profile.base_seed,
        prop_subsample=3000,
        env_prefix=env_prefix(dataset_prefix, profile),
        beta_sampler=make_beta_sampler(profile),
        response_surface=make_response_surface(profile),
        propensity_gamma_scale=profile.propensity_gamma_scale,
        done_message=f"All {dataset_prefix} {profile.name} stability streams generated successfully.",
    )


def parse_stability_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=[p.name for p in PROFILES],
        default=[p.name for p in PROFILES],
        help="Profiles to generate. Default: target good_source bad_source.",
    )
    return parser.parse_args()


def run_stability_generation(
    dataset_prefix: str,
    data_path: str,
    repeats: int,
    profile_names: Iterable[str],
) -> None:
    selected = set(profile_names)
    for profile in PROFILES:
        if profile.name not in selected:
            continue
        run_generation(build_config(dataset_prefix, data_path, profile, repeats))
