"""Shared MCMC configuration for beta-GLMM runners."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from src.models.hbb.model import resolve_pymc_ncores


@dataclass(frozen=True)
class MCMCConfig:
    n_chains: int = 2
    n_tune: int = 100
    n_samples: int = 200
    target_accept: float = 0.95
    max_treedepth: int = 8
    ncores: int | None = None
    mp_ctx: str = "spawn"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_ncores(self) -> int:
        return resolve_pymc_ncores(ncores=self.ncores, n_chains=self.n_chains)

    def format_summary(self) -> str:
        return (
            f"{self.n_chains} chains × "
            f"({self.n_tune} tune + {self.n_samples} draws)  |  "
            f"target_accept={self.target_accept}  |  "
            f"max_treedepth={self.max_treedepth}  |  "
            f"ncores={self.resolve_ncores()}  |  "
            f"mp_ctx={self.mp_ctx}"
        )


CV_MCMC_DEFAULTS = MCMCConfig()

INVESTIGATION_MCMC_DEFAULTS = MCMCConfig(
    n_chains=6,
    n_tune=10_000,
    n_samples=1_000,
    max_treedepth=15,
    ncores=6,
)


def apply_cv_smoke_overrides(
    *,
    output_dir: Path,
    regimes: list[str],
    models: list[str],
    mcmc: MCMCConfig,
    beta_min_train_rows: int,
    baseline_n_iter: int,
) -> tuple[Path, list[str], list[str], MCMCConfig, int, int]:
    """Apply fast settings when ``RCV_SMOKE=1``."""
    if os.getenv("RCV_SMOKE") != "1":
        return output_dir, regimes, models, mcmc, beta_min_train_rows, baseline_n_iter

    smoke_mcmc = MCMCConfig(
        n_chains=2,
        n_tune=50,
        n_samples=50,
        target_accept=mcmc.target_accept,
        max_treedepth=mcmc.max_treedepth,
        ncores=1,
        mp_ctx=mcmc.mp_ctx,
    )
    smoke_models = models if models != ["baselines", "beta_glmm"] else ["beta_glmm"]
    return (
        output_dir / "smoke",
        regimes or ["forward_repeat_sites"],
        smoke_models,
        smoke_mcmc,
        min(beta_min_train_rows, 200),
        min(baseline_n_iter, 2),
    )


def apply_investigation_smoke(mcmc: MCMCConfig) -> MCMCConfig:
    """Short MCMC for investigation smoke tests."""
    return MCMCConfig(
        n_chains=int(os.getenv("PY_INV_SMOKE_CHAINS", "2")),
        n_tune=int(os.getenv("PY_INV_SMOKE_TUNE", "5")),
        n_samples=int(os.getenv("PY_INV_SMOKE_DRAWS", "15")),
        target_accept=mcmc.target_accept,
        max_treedepth=mcmc.max_treedepth,
        ncores=min(mcmc.ncores or 2, int(os.getenv("PY_INV_SMOKE_CHAINS", "2"))),
        mp_ctx=mcmc.mp_ctx,
    )


def add_cv_mcmc_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--beta-*`` MCMC flags on a cross-validation CLI parser."""
    defaults = CV_MCMC_DEFAULTS
    parser.add_argument(
        "--beta-n-chains",
        type=int,
        default=defaults.n_chains,
        help="Number of MCMC chains (default: %(default)s)",
    )
    parser.add_argument(
        "--beta-n-tune",
        type=int,
        default=defaults.n_tune,
        help="Number of tuning samples per chain (default: %(default)s)",
    )
    parser.add_argument(
        "--beta-n-samples",
        type=int,
        default=defaults.n_samples,
        help="Number of posterior draws per chain (default: %(default)s)",
    )
    parser.add_argument(
        "--beta-target-accept",
        type=float,
        default=defaults.target_accept,
        help="NUTS target acceptance rate (default: %(default)s)",
    )
    parser.add_argument(
        "--beta-max-treedepth",
        type=int,
        default=defaults.max_treedepth,
        help="NUTS maximum tree depth (default: %(default)s)",
    )
    parser.add_argument(
        "--beta-ncores",
        type=int,
        default=None,
        help=(
            "Parallel PyMC chain workers (default: min(--beta-n-chains, cpu_count); "
            "uses mp_ctx=spawn)"
        ),
    )


def mcmc_config_from_cv_args(args: argparse.Namespace) -> MCMCConfig:
    return MCMCConfig(
        n_chains=args.beta_n_chains,
        n_tune=args.beta_n_tune,
        n_samples=args.beta_n_samples,
        target_accept=args.beta_target_accept,
        max_treedepth=args.beta_max_treedepth,
        ncores=args.beta_ncores,
        mp_ctx=CV_MCMC_DEFAULTS.mp_ctx,
    )


def merge_mcmc_dict(config: MCMCConfig, overrides: dict[str, Any] | None) -> MCMCConfig:
    if not overrides:
        return config
    valid = {f.name for f in fields(MCMCConfig)}
    return MCMCConfig(**{**config.to_dict(), **{k: v for k, v in overrides.items() if k in valid}})
