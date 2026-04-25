import numpy as np
from sklearn.metrics import mean_squared_error


def compute_metrics(predictions, ground_truth):
    """Compute RMSE and accuracy metrics with robust shape alignment."""

    if predictions is None or ground_truth is None:
        return {"rmse": float("inf"), "mae": float("inf"), "correlation": 0, "accuracy": 0}

    # Convert to numpy arrays
    pred = np.asarray(predictions)
    truth = np.asarray(ground_truth)

    # Ensure we have at least (time, ny, nx)
    if pred.ndim < 2 or truth.ndim < 2:
        return {"rmse": float("inf"), "mae": float("inf"), "correlation": 0, "accuracy": 0}

    # If spatial shapes differ -> can't compare directly
    # Try to support pred with an extra leading dimension (e.g., ensemble or cascade)
    # Cases to handle:
    #  - pred.shape == truth.shape -> fine
    #  - pred has shape (M, T, ny, nx) while truth is (T, ny, nx) and M>1 -> average over M
    #  - pred has shape (K, ny, nx) while truth is (T, ny, nx) -> handle if K is multiple of T or K>=T

    # Normalize to (T, ny, nx)
    if pred.shape == truth.shape:
        aligned_pred = pred
    else:
        # If pred has one extra leading dim but same spatial dims
        if pred.ndim == truth.ndim + 1 and pred.shape[1:] == truth.shape:
            # pred is (M, T, ny, nx) -> average over M to get (T, ny, nx)
            M = pred.shape[0]
            print(f"  Metrics: averaging {M} ensemble members")
            aligned_pred = np.mean(pred, axis=0)
        elif pred.ndim == truth.ndim and pred.shape[1:] == truth.shape[1:]:
            # pred is (K, ny, nx) and truth is (T, ny, nx)
            K = pred.shape[0]
            T = truth.shape[0]
            if K % T == 0:
                # e.g. K = groups * T -> reshape and average over groups
                groups = K // T
                try:
                    aligned_pred = pred.reshape(groups, T, *pred.shape[1:]).mean(axis=0)
                except Exception:
                    # fallback: take first T frames
                    aligned_pred = pred[:T]
            elif K >= T:
                # take first T frames (most conservative)
                aligned_pred = pred[:T]
            else:
                raise ValueError(f"Predictions have fewer timesteps ({K}) than ground truth ({T}).")
        else:
            # Shapes incompatible
            raise ValueError(f"Incompatible shapes: predictions {pred.shape}, ground_truth {truth.shape}")

    # Now aligned_pred and truth should have the same shape
    if aligned_pred.shape != truth.shape:
        raise ValueError(f"Failed to align shapes: aligned_pred {aligned_pred.shape}, truth {truth.shape}")

    # Flatten and compute metrics, excluding non-finite values
    pred_flat = aligned_pred.flatten()
    truth_flat = truth.flatten()

    valid_mask = np.isfinite(pred_flat) & np.isfinite(truth_flat)
    pred_valid = pred_flat[valid_mask]
    truth_valid = truth_flat[valid_mask]

    if pred_valid.size == 0:
        return {"rmse": float("inf"), "mae": float("inf"), "correlation": 0, "accuracy": 0}

    rmse = np.sqrt(mean_squared_error(truth_valid, pred_valid))
    mae = np.mean(np.abs(pred_valid - truth_valid))

    if np.std(pred_valid) > 0 and np.std(truth_valid) > 0:
        correlation = np.corrcoef(pred_valid, truth_valid)[0, 1]
    else:
        correlation = 0.0

    relative_error = np.abs(pred_valid - truth_valid) / (np.abs(truth_valid) + 1e-6)
    accuracy = float(np.mean(relative_error < 0.2) * 100.0)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "accuracy": accuracy,
        "valid_points": int(pred_valid.size),
        "total_points": int(pred_flat.size),
    }


def print_comparison(linda_results, pinn_results):
    """Print comparison of results"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    # Compute metrics
    linda_metrics = compute_metrics(linda_results["predictions"], linda_results["ground_truth"])
    pinn_metrics = compute_metrics(pinn_results["predictions"], pinn_results["ground_truth"])

    # Print results
    print(f"\n{linda_results['model_name']}:")
    print(f"  RMSE: {linda_metrics['rmse']:.4f}")
    print(f"  MAE: {linda_metrics['mae']:.4f}")
    print(f"  Correlation: {linda_metrics['correlation']:.4f}")
    print(f"  Accuracy (±20%): {linda_metrics['accuracy']:.2f}%")
    print(f"  Valid points: {linda_metrics['valid_points']}/{linda_metrics['total_points']}")

    print(f"\n{pinn_results['model_name']}:")
    print(f"  RMSE: {pinn_metrics['rmse']:.4f}")
    print(f"  MAE: {pinn_metrics['mae']:.4f}")
    print(f"  Correlation: {pinn_metrics['correlation']:.4f}")
    print(f"  Accuracy (±20%): {pinn_metrics['accuracy']:.2f}%")
    print(f"  Valid points: {pinn_metrics['valid_points']}/{pinn_metrics['total_points']}")

    if "training_time" in pinn_results:
        print(f"  Training time: {pinn_results['training_time']:.2f}s")

    # Determine winner
    print(f"\n{'=' * 60}")
    print("SUMMARY:")

    metrics_comparison = []
    if linda_metrics["rmse"] < pinn_metrics["rmse"]:
        metrics_comparison.append(f"RMSE: {linda_results['model_name']} wins")
    elif pinn_metrics["rmse"] < linda_metrics["rmse"]:
        metrics_comparison.append(f"RMSE: {pinn_results['model_name']} wins")
    else:
        metrics_comparison.append("RMSE: Tie")

    if linda_metrics["accuracy"] > pinn_metrics["accuracy"]:
        metrics_comparison.append(f"Accuracy: {linda_results['model_name']} wins")
    elif pinn_metrics["accuracy"] > linda_metrics["accuracy"]:
        metrics_comparison.append(f"Accuracy: {pinn_results['model_name']} wins")
    else:
        metrics_comparison.append("Accuracy: Tie")

    for comparison in metrics_comparison:
        print(f"  {comparison}")

    print("=" * 60)
