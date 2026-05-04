from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.monte_carlo import ExperimentConfig, run_monte_carlo, summarize_results
from models.neural_networks import default_torch_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run Appendix A Monte Carlo simulations.")
    parser.add_argument("--cases", nargs="+", default=["a"], choices=["a", "b", "c"])
    parser.add_argument("--pcs", nargs="+", default=[50], type=int)
    parser.add_argument("--repetitions", default=1, type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["OLS-3", "ENet", "RF"],
        choices=["OLS-3", "OLS-3-H", "ENet", "ENet-H", "RF", "NN1", "NN2", "NN3"],
    )
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--rf-trees", default=300, type=int)
    parser.add_argument("--nn-epochs", default=100, type=int)
    parser.add_argument("--nn-ensemble-size", default=10, type=int)
    parser.add_argument("--nn-tune-ensemble-size", default=None, type=int)
    parser.add_argument("--nn-batch-size", default=10_000, type=int)
    parser.add_argument("--nn-device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Use a small grid for a fast smoke test.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--resume", action="store_true", help="Resume from an existing raw results CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        experiment = ExperimentConfig(
            cases=args.cases,
            pc_values=args.pcs,
            repetitions=args.repetitions,
            models=args.models,
            seed=args.seed,
            enet_alphas=(1e-4, 1e-3, 1e-2),
            rf_depths=(1, 2),
            rf_max_features=(3, 10),
            rf_n_estimators=min(args.rf_trees, 50),
            nn_learning_rates=(0.001,),
            nn_l1_penalties=(1e-5,),
            nn_epochs=min(args.nn_epochs, 25),
            nn_patience=3,
            nn_ensemble_size=min(args.nn_ensemble_size, 2),
            nn_tune_ensemble_size=args.nn_tune_ensemble_size,
            nn_batch_size=args.nn_batch_size,
            nn_device=args.nn_device,
        )
    else:
        experiment = ExperimentConfig(
            cases=args.cases,
            pc_values=args.pcs,
            repetitions=args.repetitions,
            models=args.models,
            seed=args.seed,
            rf_n_estimators=args.rf_trees,
            nn_epochs=args.nn_epochs,
            nn_ensemble_size=args.nn_ensemble_size,
            nn_tune_ensemble_size=args.nn_tune_ensemble_size,
            nn_batch_size=args.nn_batch_size,
            nn_device=args.nn_device,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(model.startswith("NN") for model in args.models):
        print(f"Using torch device for neural networks: {args.nn_device or default_torch_device()}")

    raw_path = output_dir / "monte_carlo_raw_results.csv"
    summary_path = output_dir / "monte_carlo_summary.csv"
    results = run_monte_carlo(experiment, checkpoint_path=raw_path, resume=args.resume)
    summary = summarize_results(results)

    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
