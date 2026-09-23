import argparse
import logging
import sys
from pathlib import Path

import optuna
import torch

from msprep import MultiStatePrep
from experiments.nestedcv import NestedCrossVal
from experiments.config import MetaConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested CV with hyperparameter search.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to MetaConfig YAML.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def setup_logging(output_dir: Path, experiment_name: str, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{experiment_name}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    optuna.logging.set_verbosity(level)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()


def main() -> None:
    args = parse_args()

    # Load meta config
    meta_cfg = MetaConfig.from_yaml(args.config)

    # Apply simple overrides
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed

    # device override / auto-detection
    if args.device is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("device='cuda' requested but not available, falling back to cpu.", file=sys.stderr)
            dev = "cpu"
        else:
            dev = args.device
    overrides["device"] = dev

    meta_cfg = meta_cfg.with_updates(**overrides)

    # Load dataset from config
    msp = MultiStatePrep.load(meta_cfg.dataset_path)


    # Compute experiment_name and output_dir if not set
    if meta_cfg.experiment_name == "default":
        experiment_name = f"{Path(meta_cfg.dataset_path).stem}"
    else:
        experiment_name = meta_cfg.experiment_name

    output_dir = Path(meta_cfg.save_dir or "results") / experiment_name
    meta_cfg = meta_cfg.with_updates(
        experiment_name=experiment_name,
        save_dir=str(output_dir),
    )

    setup_logging(output_dir, experiment_name)
    logger = logging.getLogger(__name__)

    logger.info("ExperimentConfig: %s", meta_cfg.to_dict())
    meta_cfg.to_yaml(output_dir / "config.yaml")

    # Run nested CV
    runner = NestedCrossVal(
        msp=msp,
        meta_cfg=meta_cfg,
        log=logger,
    )
    _ = runner.run()
    logger.info("Experiment finished.")


if __name__ == "__main__":
    main()