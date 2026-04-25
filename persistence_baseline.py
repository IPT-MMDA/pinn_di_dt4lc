"""
Persistence baseline comparison script.

Runs LINDA, LINDA-PINN, and a naïve persistence baseline on the same data,
then prints metrics and saves a 4-row comparison plot.

Usage:
    uv run persistence_baseline.py # synthetic data
    uv run persistence_baseline.py --real # Swiss radar data
    uv run persistence_baseline.py --out fig.png
"""

import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from app import (
    compute_metrics,
    generate_synthetic_data,
    load_swiss_radar_data,
    train_custom_pinn_with_params,
    train_traditional_linda_with_params,
)


# Persistence baseline
def predict_persistence(rainrate_sequence, n_input, n_forecast):
    """Naive baseline: repeat the last observed frame for every forecast step."""
    last_frame = rainrate_sequence[n_input - 1]
    predictions = np.tile(last_frame[np.newaxis, :, :], (n_forecast, 1, 1))
    ground_truth = rainrate_sequence[n_input : n_input + n_forecast]
    return {
        "model_name": "Persistence",
        "predictions": predictions, # (n_forecast, ny, nx)
        "ground_truth": ground_truth, # (n_forecast, ny, nx)
    }


# Visualization
def create_comparison_plot(linda_results, pinn_results, persistence_results, max_frames=6):
    """4-row comparison: Ground Truth / LINDA / PINN / Persistence."""
    linda_pred = linda_results["predictions"]
    pinn_pred = pinn_results["predictions"]
    persist_pred = persistence_results["predictions"]
    ground_truth = linda_results["ground_truth"]

    if linda_pred.ndim == 4: # average ensemble members
        linda_pred = np.mean(linda_pred, axis=0)

    n_frames = min(
        max_frames,
        ground_truth.shape[0],
        linda_pred.shape[0],
        pinn_pred.shape[0],
        persist_pred.shape[0],
    )

    fig = plt.figure(figsize=(n_frames * 3 + 1, 13))
    gs = gridspec.GridSpec(4, n_frames + 1, width_ratios=[1] * n_frames + [0.05], hspace=0.3, wspace=0.2)

    vmin = 0
    vmax = max(
        np.max(ground_truth[:n_frames]),
        np.max(linda_pred[:n_frames]),
        np.max(pinn_pred[:n_frames]),
        np.max(persist_pred[:n_frames]),
    )

    rows = [
        ("Truth", ground_truth),
        ("LINDA", linda_pred),
        ("PINN", pinn_pred),
        ("Persistence", persist_pred),
    ]
    first_im = None
    for row_idx, (label, data) in enumerate(rows):
        for t in range(n_frames):
            ax = fig.add_subplot(gs[row_idx, t])
            frame = data[t] if t < len(data) else np.zeros_like(ground_truth[0])
            im = ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} t+{t + 1}", fontsize=9)
            ax.axis("off")
            if first_im is None:
                first_im = im

    row_y = [1 - (i + 0.5) / 4 for i in range(4)]
    for label, y in zip(["Ground Truth", "LINDA", "PINN", "Persistence"], row_y):
        fig.text(0.02, y, label, rotation=90, verticalalignment="center", fontsize=11, weight="bold")

    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(first_im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Precipitation (mm/h)", rotation=270, labelpad=20)

    plt.suptitle("Nowcasting Comparison — with Persistence Baseline", fontsize=13, y=0.98)
    return fig


# Reporting
def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def print_metrics(name, metrics):
    print(f"\n  {name}")
    print(f"    RMSE:         {_fmt(metrics['rmse'])}")
    print(f"    MAE:          {_fmt(metrics['mae'])}")
    print(f"    Correlation:  {_fmt(metrics['correlation'])}")
    print(f"    Accuracy±20%: {metrics['accuracy']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Persistence baseline comparison")
    parser.add_argument("--real", action="store_true", help="Use Swiss radar data instead of synthetic")
    parser.add_argument("--n-input", type=int, default=3)
    parser.add_argument("--n-forecast", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="persistence_comparison.png", help="Output figure path")
    args = parser.parse_args()

    # Load data
    if args.real:
        try:
            rainrate_sequence, metadata = load_swiss_radar_data()
            print("Loaded Swiss radar data.")
        except Exception as e:
            print(f"Failed to load real data ({e}), falling back to synthetic.")
            rainrate_sequence, metadata = generate_synthetic_data()
    else:
        rainrate_sequence, metadata = generate_synthetic_data()
        print("Using synthetic data.")

    n_input, n_forecast = args.n_input, args.n_forecast

    # Persistence
    print("\nComputing persistence baseline...")
    persistence_results = predict_persistence(rainrate_sequence, n_input, n_forecast)

    # LINDA
    print("\nTraining LINDA...")
    linda_results = train_traditional_linda_with_params(
        rainrate_sequence, metadata,
        n_input=n_input,
        n_forecast=n_forecast,
    )

    # PINN
    print("\nTraining PINN...")
    pinn_results = train_custom_pinn_with_params(
        rainrate_sequence, metadata,
        n_input=n_input,
        n_forecast=n_forecast,
        epochs=args.epochs,
    )

    # Metrics
    linda_metrics       = compute_metrics(linda_results["predictions"],       linda_results["ground_truth"])
    pinn_metrics        = compute_metrics(pinn_results["predictions"],        pinn_results["ground_truth"])
    persistence_metrics = compute_metrics(persistence_results["predictions"], persistence_results["ground_truth"])

    print("\n=== Metrics ===")
    print_metrics("Persistence (baseline)", persistence_metrics)
    print_metrics("Traditional LINDA",      linda_metrics)
    print_metrics("LINDA-PINN",             pinn_metrics)

    # Winner
    scores = {"Persistence": persistence_metrics["rmse"], "LINDA": linda_metrics["rmse"], "PINN": pinn_metrics["rmse"]}
    winner = min(scores, key=scores.get)
    print(f"\nBest RMSE: {winner} ({scores[winner]:.4f})")

    # Plot
    fig = create_comparison_plot(linda_results, pinn_results, persistence_results)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nFigure saved to {args.out}")


if __name__ == "__main__":
    main()
