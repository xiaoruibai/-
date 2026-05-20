#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Iterable

import numpy as np

from stream_generation_common import StreamGenerationConfig, run_generation


@dataclass(frozen=True)
class ModerateProfile:
    name: str
    env_prefix_suffix: str
    outdir_suffix: str
    t: int
    base_seed: int
    propensity_gamma_scale: float
    beta_values: tuple[float, ...]
    beta_probs: tuple[float, ...]
    beta_jitter: float
    quad_scale: float
    base: float
    tau_base: float
    a0: float
    a1: float
    b0: float
    b1: float
    offset0: float
    offset1: float
    sign_flip_rate: float = 0.0


TARGET_PROFILE = ModerateProfile(
    name="target",
    env_prefix_suffix="",
    outdir_suffix="moderate_streams_repeats50_target",
    t=1000,
    base_seed=2026,
    propensity_gamma_scale=1.0,
    beta_values=(0.0, 0.08, 0.16, 0.24, 0.32),
    beta_probs=(0.36, 0.20, 0.20, 0.14, 0.10),
    beta_jitter=0.01,
    quad_scale=0.35,
    base=0.08,
    tau_base=0.20,
    a0=0.50,
    a1=0.34,
    b0=0.18,
    b1=0.12,
    offset0=0.02,
    offset1=0.06,
)

GOOD_SOURCE_PROFILE = ModerateProfile(
    name="good_source",
    env_prefix_suffix="good",
    outdir_suffix="moderate_streams_repeats50_source",
    t=3000,
    base_seed=2027,
    propensity_gamma_scale=1.10,
    beta_values=(0.0, 0.075, 0.15, 0.22, 0.30),
    beta_probs=(0.35, 0.21, 0.20, 0.14, 0.10),
    beta_jitter=0.015,
    quad_scale=0.32,
    base=0.10,
    tau_base=0.19,
    a0=0.47,
    a1=0.32,
    b0=0.16,
    b1=0.11,
    offset0=0.035,
    offset1=0.075,
)

BAD_SOURCE_PROFILE = ModerateProfile(
    name="bad_source",
    env_prefix_suffix="bad",
    outdir_suffix="bad_moderate_streams_repeats50_source",
    t=3000,
    base_seed=520,
    propensity_gamma_scale=1.35,
    beta_values=(0.0, -0.08, -0.15, -0.22, 0.12),
    beta_probs=(0.28, 0.24, 0.22, 0.16, 0.10),
    beta_jitter=0.025,
    quad_scale=-0.45,
    base=-0.02,
    tau_base=0.02,
    a0=-0.42,
    a1=-0.24,
    b0=0.26,
    b1=-0.16,
    offset0=-0.12,
    offset1=0.18,
    sign_flip_rate=0.30,
)

PROFILES = (TARGET_PROFILE, GOOD_SOURCE_PROFILE, BAD_SOURCE_PROFILE)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v
    return v / norm


def make_beta_sampler(profile: ModerateProfile):
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
        beta_quad = profile.quad_scale * _normalize(beta + rng.normal(0.0, 0.025, size=d))
        return np.concatenate([beta, beta_quad]).astype(float)

    return sample_beta


def make_response_surface(profile: ModerateProfile):
    def response_surface(X: np.ndarray, packed_beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = X.shape[1]
        beta = packed_beta[:d]
        beta_quad = packed_beta[d: 2 * d]
        score = X @ beta
        quad = (X ** 2) @ beta_quad
        mu_m1 = (
            profile.base
            + profile.a0 * np.tanh(score + profile.offset0)
            + profile.b0 * quad
        )
        mu_p1 = (
            mu_m1
            + profile.tau_base
            + profile.a1 * np.tanh(score + profile.offset1)
            + profile.b1 * quad
        )
        return mu_m1, mu_p1

    return response_surface


def env_prefix(dataset_prefix: str, profile: ModerateProfile) -> str:
    if not profile.env_prefix_suffix:
        return dataset_prefix
    return f"{dataset_prefix}_{profile.env_prefix_suffix}"


def outdir(dataset_prefix: str, profile: ModerateProfile) -> str:
    return f"{dataset_prefix}_{profile.outdir_suffix}"


def build_config(
    dataset_prefix: str,
    data_path: str,
    profile: ModerateProfile,
    repeats: int,
) -> StreamGenerationConfig:
    return StreamGenerationConfig(
        dataset_prefix=dataset_prefix,
        data_path=data_path,
        outdir=outdir(dataset_prefix, profile),
        t=profile.t,
        repeats=repeats,
        eps=0.05,
        noise_var=0.1,
        base_seed=profile.base_seed,
        prop_subsample=3000,
        env_prefix=env_prefix(dataset_prefix, profile),
        beta_sampler=make_beta_sampler(profile),
        response_surface=make_response_surface(profile),
        propensity_gamma_scale=profile.propensity_gamma_scale,
        done_message=f"All {dataset_prefix} {profile.name} moderate streams generated successfully.",
    )


def parse_moderate_args(description: str) -> argparse.Namespace:
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


def run_moderate_generation(
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
