import numpy as np
import matplotlib.pyplot as plt


def create_prediction_visualization(linda_results, pinn_results, max_frames=6):
    """Create side-by-side visualization of predictions with better colorbar placement"""

    # Get predictions and ground truth
    linda_pred = linda_results["predictions"]
    pinn_pred = pinn_results["predictions"]
    ground_truth = linda_results["ground_truth"]

    # Handle ensemble predictions: average over ensemble members
    if linda_pred.ndim == 4:  # (ensemble, time, ny, nx)
        print(f"  LINDA: averaging {linda_pred.shape[0]} ensemble members")
        linda_pred = np.mean(linda_pred, axis=0)

    # Determine number of frames to show
    n_frames = min(max_frames, ground_truth.shape[0], linda_pred.shape[0], pinn_pred.shape[0])

    # Create figure with subplots - add space for colorbar
    fig = plt.figure(figsize=(n_frames * 3 + 1, 10))  # Extra width for colorbar

    # Create grid spec for better control
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(3, n_frames + 1, width_ratios=[1] * n_frames + [0.05], hspace=0.3, wspace=0.2)

    vmin = 0
    vmax = max(np.max(ground_truth[:n_frames]), np.max(linda_pred[:n_frames]), np.max(pinn_pred[:n_frames]))

    # Store all image mappables for colorbar
    images = []

    for t in range(n_frames):
        # Ground truth
        ax1 = fig.add_subplot(gs[0, t])
        im1 = ax1.imshow(ground_truth[t], cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Truth t+{t + 1}", fontsize=10)
        ax1.axis("off")
        images.append(im1)

        # LINDA prediction
        ax2 = fig.add_subplot(gs[1, t])
        im2 = ax2.imshow(
            linda_pred[t] if t < len(linda_pred) else np.zeros_like(ground_truth[0]),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax2.set_title(f"LINDA t+{t + 1}", fontsize=10)
        ax2.axis("off")

        # PINN prediction
        ax3 = fig.add_subplot(gs[2, t])
        im3 = ax3.imshow(
            pinn_pred[t] if t < len(pinn_pred) else np.zeros_like(ground_truth[0]),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax3.set_title(f"PINN t+{t + 1}", fontsize=10)
        ax3.axis("off")

    # Add row labels
    fig.text(0.02, 0.75, "Ground Truth", rotation=90, verticalalignment="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.5, "LINDA", rotation=90, verticalalignment="center", fontsize=12, weight="bold")
    fig.text(0.02, 0.25, "PINN", rotation=90, verticalalignment="center", fontsize=12, weight="bold")

    # Add single colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(images[0], cax=cbar_ax, orientation="vertical")
    cbar.set_label("Precipitation (mm/h)", rotation=270, labelpad=20)

    plt.suptitle("Precipitation Nowcasting Comparison", fontsize=14, y=0.98)

    return fig


def create_loss_plot(pinn_results):
    """Create loss evolution plot for PINN"""
    if "losses" not in pinn_results or len(pinn_results["losses"]) == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    ax1.plot(pinn_results["losses"], label="Total Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("PINN Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Physics loss
    if "physics_losses" in pinn_results and len(pinn_results["physics_losses"]) > 0:
        ax2.plot(pinn_results["physics_losses"], label="Physics Loss", linewidth=2, color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Physics Loss")
        ax2.set_title("Physics-Informed Loss")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    return fig
