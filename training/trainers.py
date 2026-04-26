import time

import numpy as np
import torch
import torch.nn.functional as F
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts import linda as pysteps_linda
from pysteps.utils import conversion

from data.loaders import generate_synthetic_data, load_swiss_radar_data
from evaluation.metrics import compute_metrics
from evaluation.visualization import create_loss_plot, create_prediction_visualization
from models.linda_pinn import LINDAPINNModel, device


class LINDAPINNTrainer:
    def __init__(self, spatial_domain=(-100, 100), temporal_domain=(0, 6)):
        self.model = LINDAPINNModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100)

        self.x_min, self.x_max = spatial_domain
        self.t_min, self.t_max = temporal_domain

        # Print device info
        print(f"Model initialized on device: {device}")
        print(f"Model parameters on device: {next(self.model.parameters()).device}")

    def prepare_training_data_from_radar(self, rainrate_sequence, metadata, advection_field=None):
        """Convert radar sequence to training data for LINDA-PINN

        Computes advection field using dense_lucaskanade if not provided.
        """
        nt, ny, nx = rainrate_sequence.shape

        # Compute advection field if not provided
        if advection_field is None:
            print("Computing advection field from radar sequence...")
            # Use first 3 frames to compute motion field (same as traditional LINDA)
            n_input = min(3, nt)
            R_input = rainrate_sequence[:n_input]

            # Convert to rainrate if needed
            try:
                conv_out = conversion.to_rainrate(R_input, metadata)
                if isinstance(conv_out, tuple) and len(conv_out) >= 1:
                    conv_arr = conv_out[0]
                else:
                    conv_arr = conv_out

                if isinstance(conv_arr, (list, tuple)):
                    arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr]
                    R_input_rr = np.stack(arrs, axis=0)
                elif isinstance(conv_arr, np.ndarray) and conv_arr.dtype == object:
                    arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr.tolist()]
                    R_input_rr = np.stack(arrs, axis=0)
                else:
                    R_input_rr = np.asarray(conv_arr, dtype=np.float32)

                # Remove NaN/inf values
                finite_mask = np.isfinite(R_input_rr)
                if not finite_mask.all():
                    R_input_rr[~finite_mask] = 0.0

                # Compute motion field
                advection_field = dense_lucaskanade(R_input_rr)
                print(f"Advection field computed: shape {advection_field.shape}")
            except Exception as e:
                print(f"Failed to compute advection field: {e}")
                advection_field = None

        # Prepare coordinate grids
        if "xpixelsize" in metadata and "ypixelsize" in metadata:
            x_coords = np.arange(nx) * metadata["xpixelsize"] / 1000.0
            y_coords = np.arange(ny) * metadata["ypixelsize"] / 1000.0
        else:
            x_coords = np.linspace(self.x_min, self.x_max, nx)
            y_coords = np.linspace(self.x_min, self.x_max, ny)

        training_pairs = []

        for t in range(nt - 1):
            R_current = rainrate_sequence[t]
            R_next = rainrate_sequence[t + 1]

            # Improved masking: include rainy pixels AND transition regions
            # This allows learning dry→rain and rain→dry transitions
            rainy_mask = R_current > 0.1
            next_rainy_mask = R_next > 0.1

            # Include: (1) current rain, (2) future rain, (3) transition zones
            # Use dilation to capture edges of rain regions
            from scipy.ndimage import binary_dilation

            # Dilate rainy regions to include boundary/transition pixels
            rainy_dilated = binary_dilation(rainy_mask, iterations=2)
            next_rainy_dilated = binary_dilation(next_rainy_mask, iterations=2)

            # Combine: rain now, rain later, or near rain (transition zones)
            mask = rainy_mask | next_rainy_mask | rainy_dilated | next_rainy_dilated

            # Ensure we have enough pixels for training
            if np.sum(mask) > 100:
                training_pairs.append(
                    {
                        "R_current": torch.tensor(R_current, dtype=torch.float32, device=device),
                        "R_next": torch.tensor(R_next, dtype=torch.float32, device=device),
                        "x_coords": x_coords,
                        "y_coords": y_coords,
                        "t": float(t),
                        "mask": torch.tensor(mask, dtype=torch.bool, device=device),
                        "advection": advection_field,
                        "metadata": metadata,
                    }
                )

        return training_pairs

    def compute_physics_loss(self, training_pair):
        """Compute physics-informed loss for proper LINDA IDE"""
        R_current = training_pair["R_current"]
        R_target = training_pair["R_next"]
        x_coords = training_pair["x_coords"]
        y_coords = training_pair["y_coords"]
        t = training_pair["t"]
        mask = training_pair["mask"]
        advection = training_pair.get("advection", None)
        metadata = training_pair.get("metadata", {})

        # Forward pass through model
        R_predicted = self.model(R_current, x_coords, y_coords, t, advection)

        # 1. Data loss (supervised)
        if torch.sum(mask) > 0:
            data_loss = F.mse_loss(R_predicted[mask], R_target[mask])
        else:
            data_loss = torch.tensor(0.0, device=device)

        # 2. Physics loss - enforce IDE structure (aligned with forward pass)
        with torch.enable_grad():
            # Recompute advection and integral as in forward pass
            if advection is not None:
                R_advected_physics = self.model.apply_advection(R_current, advection, metadata)
            else:
                R_advected_physics = R_current

            # Integral term on advected field (matches forward pass)
            integral_term = self.model.compute_integral_term(R_advected_physics, x_coords, y_coords, t, metadata)

            # Dispersal conservation (check integral on advected field)
            total_before = torch.sum(R_advected_physics)
            total_integral = torch.sum(integral_term)
            dispersal_conservation = torch.abs(total_integral - total_before) / (total_before + 1e-6)

            # Growth bounds (ensure realistic growth on advected field)
            growth_rate = torch.sigmoid(self.model.growth_rate)
            max_growth = growth_rate * R_advected_physics * (1 - R_advected_physics / self.model.carrying_capacity)
            growth_penalty = torch.mean(F.relu(max_growth - 0.5))  # Penalize excessive growth

            # Advection conservation
            if advection is not None:
                advection_diff = torch.mean(torch.abs(torch.sum(R_advected_physics) - torch.sum(R_current)))
            else:
                advection_diff = torch.tensor(0.0, device=device)

        # 3. Smoothness regularization
        if R_predicted.shape[0] > 1 and R_predicted.shape[1] > 1:
            grad_x = torch.diff(R_predicted, dim=1)
            grad_y = torch.diff(R_predicted, dim=0)
            smoothness_loss = torch.mean(grad_x**2) + torch.mean(grad_y**2)
        else:
            smoothness_loss = torch.tensor(0.0, device=device)

        # 4. Parameter regularization
        param_reg = (
            torch.abs(self.model.log_sigma)  # Prevent extreme kernel widths
            + torch.abs(self.model.survival_prob - 0.8)
            + torch.abs(self.model.growth_rate - 0.1)
        )

        # Combine losses with proper weighting
        total_loss = (
            data_loss
            + 0.1 * dispersal_conservation
            + 0.05 * growth_penalty
            + 0.05 * advection_diff
            + 0.01 * smoothness_loss
            + 0.01 * param_reg
        )

        return total_loss, {
            "data_loss": data_loss.item(),
            "dispersal_conservation": dispersal_conservation.item(),
            "growth_penalty": growth_penalty.item(),
            "advection_diff": advection_diff.item(),
            "smoothness_loss": smoothness_loss.item(),
        }

    def train_on_radar_sequence(self, rainrate_sequence, metadata, epochs=10, verbose=True):
        """Train PINN on radar data sequence"""
        training_data = self.prepare_training_data_from_radar(rainrate_sequence, metadata)

        if len(training_data) == 0:
            raise ValueError("No valid training data found!")

        print(f"Created {len(training_data)} training pairs")
        print(f"Training on device: {device}")

        losses = []
        physics_losses = []
        loss_components = {
            "data_loss": [],
            "dispersal_conservation": [],
            "growth_penalty": [],
            "advection_diff": [],
            "smoothness_loss": [],
        }

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_physics_loss = 0
            epoch_components = {k: 0 for k in loss_components.keys()}
            valid_batches = 0

            np.random.shuffle(training_data)

            for training_pair in training_data:
                self.optimizer.zero_grad()

                loss_output = self.compute_physics_loss(training_pair)

                # Handle different return types
                if isinstance(loss_output, tuple) and len(loss_output) == 2:
                    loss, loss_details = loss_output
                    # Extract physics loss from the dictionary
                    physics_loss = loss_details.get(
                        "data_loss", 0.0
                    )  # NOTE: It's actually physics loss, naming misaligned

                    # Accumulate component losses
                    for key, value in loss_details.items():
                        if key in epoch_components:
                            epoch_components[key] += value
                else:
                    # Fallback for old format
                    loss = loss_output
                    physics_loss = loss.item() if hasattr(loss, "item") else 0.0

                if loss.requires_grad and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_physics_loss += physics_loss
                    valid_batches += 1

            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                avg_physics_loss = epoch_physics_loss / valid_batches

                # Average component losses
                for key in epoch_components:
                    epoch_components[key] /= valid_batches
                    loss_components[key].append(epoch_components[key])
            else:
                avg_loss = 0
                avg_physics_loss = 0
                for key in loss_components:
                    loss_components[key].append(0)

            losses.append(avg_loss)
            physics_losses.append(avg_physics_loss)

            self.scheduler.step(avg_loss)

            if verbose and epoch % 2 == 0:
                print(f"Epoch {epoch}/{epochs}:")
                print(f"  Total Loss: {avg_loss:.6f}")
                print(f"  Physics Loss: {avg_physics_loss:.6f}")

                # Print component losses
                if valid_batches > 0:
                    print(f"  Loss components:")
                    for key, value in epoch_components.items():
                        print(f"    {key}: {value:.6f}")

                print(f"  Valid batches: {valid_batches}/{len(training_data)}")
                print(
                    f"  Learned params: σ={torch.exp(self.model.log_sigma).item():.3f}, "
                    f"s={torch.sigmoid(self.model.survival_prob).item():.3f}, "
                    f"r={torch.sigmoid(self.model.growth_rate).item():.3f}"
                )
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(
                    f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                    if torch.cuda.is_available()
                    else ""
                )
                print()

        return losses, physics_losses


def train_traditional_linda(rainrate_sequence, metadata):
    """Train traditional LINDA model using pysteps"""
    print("\n=== Training Traditional LINDA ===")

    # Split data into training and testing
    n_input = 3
    n_forecast = 128

    if rainrate_sequence.shape[0] < n_input + n_forecast:
        print("Warning: Not enough timesteps for proper train/test split")
        n_forecast = min(3, rainrate_sequence.shape[0] - n_input)

    # Use first part for nowcasting setup
    R_input = rainrate_sequence[:n_input]
    R_truth = rainrate_sequence[n_input : n_input + n_forecast]

    print(f"Input shape: {R_input.shape}")
    print(f"Truth shape: {R_truth.shape}")

    conv_out = conversion.to_rainrate(R_input, metadata)

    # If conversion returned a tuple (arr, meta) handle it
    if isinstance(conv_out, tuple) and len(conv_out) >= 1:
        conv_arr = conv_out[0]
    else:
        conv_arr = conv_out

    # If conv_arr is a list/tuple of 2D arrays, stack them.
    if isinstance(conv_arr, (list, tuple)):
        # ensure all elements are numeric 2D arrays and have the same shape
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr]
        R_input_rr = np.stack(arrs, axis=0)
    elif isinstance(conv_arr, np.ndarray) and conv_arr.dtype == object:
        # object array -> try to convert each element
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr.tolist()]
        R_input_rr = np.stack(arrs, axis=0)
    else:
        # already a ndarray of numeric dtype (either 2D or 3D)
        R_input_rr = np.asarray(conv_arr, dtype=np.float32)

    # Now R_input_rr is guaranteed numeric (n,ny,nx)

    # IMPORT DEBUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print("R_input_rr shape after stack:", R_input_rr.shape, "dtype:", R_input_rr.dtype)

    # Safe NaN/infinite replacement
    finite_mask = np.isfinite(R_input_rr)
    if not finite_mask.all():
        R_input_rr[~finite_mask] = 0.0

    # Compute motion field
    print("Computing motion field...")
    motion_field = dense_lucaskanade(R_input_rr)
    print(f"Motion field shape: {motion_field.shape}")

    # Initialize LINDA
    print("Initializing LINDA...")

    # Use metadata for kmperpixel and timestep
    kmperpixel = metadata.get("xpixelsize", 1000) / 1000.0
    timestep = metadata.get("accutime", 5.0)
    print(f"  kmperpixel: {kmperpixel}, timestep: {timestep} min")

    linda_forecast = pysteps_linda.forecast(
        R_input_rr,  # 3D: (n_input, ny, nx)
        motion_field,  # (2, ny, nx)
        n_forecast,
        kmperpixel=kmperpixel,
        timestep=timestep,
        n_ens_members=10,
        vel_pert_kwargs={"p_pert_par": [1.0, 0.1, 0.01, 0.1, 0.01]},
    )

    print(f"LINDA forecast shape: {linda_forecast.shape}")

    return {
        "model_name": "Traditional LINDA",
        "predictions": linda_forecast,
        "ground_truth": R_truth,
        "metadata": metadata,
        "motion_field": motion_field,
    }


def train_custom_pinn(rainrate_sequence, metadata):
    """Train custom LINDA-PINN model"""
    print("\n=== Training Custom LINDA-PINN ===")

    # Initialize trainer
    trainer = LINDAPINNTrainer()

    # Clean train/test split - NO overlap
    n_test = 3
    n_input = 3  # Need 3 input frames for advection computation

    # Test set: last n_test frames
    test_start_idx = rainrate_sequence.shape[0] - n_test
    test_sequence = rainrate_sequence[test_start_idx:]

    # Training set: all frames BEFORE test set (no overlap)
    train_sequence = rainrate_sequence[:test_start_idx]

    # Ensure we have enough training data
    if train_sequence.shape[0] < n_input:
        raise ValueError(
            f"Not enough data for training. Need at least {n_input} frames for train, got {train_sequence.shape[0]}"
        )

    print(f"Training sequence shape: {train_sequence.shape}")
    print(f"Test sequence shape: {test_sequence.shape}")

    # Compute advection field from training data (same as traditional LINDA)
    print("Computing advection field from training data...")
    R_input = train_sequence[:n_input]
    conv_out = conversion.to_rainrate(R_input, metadata)
    if isinstance(conv_out, tuple) and len(conv_out) >= 1:
        conv_arr = conv_out[0]
    else:
        conv_arr = conv_out
    if isinstance(conv_arr, (list, tuple)):
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr]
        R_input_rr = np.stack(arrs, axis=0)
    elif isinstance(conv_arr, np.ndarray) and conv_arr.dtype == object:
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr.tolist()]
        R_input_rr = np.stack(arrs, axis=0)
    else:
        R_input_rr = np.asarray(conv_arr, dtype=np.float32)
    finite_mask = np.isfinite(R_input_rr)
    if not finite_mask.all():
        R_input_rr[~finite_mask] = 0.0
    advection_field = dense_lucaskanade(R_input_rr)
    print(f"Advection field computed: shape {advection_field.shape}")

    try:
        # Train the model
        start_time = time.time()
        losses, physics_losses = trainer.train_on_radar_sequence(train_sequence, metadata, epochs=10, verbose=True)
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Make predictions on test data (recursive forecasting)
        print("Making predictions...")
        predictions = []

        # Start with the last frame from training (last known frame before test)
        current_frame = train_sequence[-1]
        previous_frame = train_sequence[-2] if len(train_sequence) >= 2 else train_sequence[-1]

        for t in range(n_test):
            # Create coordinate grids
            ny, nx = current_frame.shape
            if "xpixelsize" in metadata and "ypixelsize" in metadata:
                x_coords = np.arange(nx) * metadata["xpixelsize"] / 1000.0
                y_coords = np.arange(ny) * metadata["ypixelsize"] / 1000.0
            else:
                x_coords = np.linspace(-100, 100, nx)
                y_coords = np.linspace(-100, 100, ny)

            # Convert to tensor
            current_tensor = torch.tensor(current_frame, dtype=torch.float32, device=device)

            # Predict next frame - PASS advection field (critical fix)
            with torch.no_grad():
                next_frame = trainer.model(current_tensor, x_coords, y_coords, float(t), advection_field, metadata)
                next_frame_np = next_frame.cpu().numpy()
                predictions.append(next_frame_np)

            # Use prediction as input for next step (recursive forecasting)
            previous_frame = current_frame
            current_frame = next_frame_np

        predictions = np.array(predictions) if predictions else np.zeros((n_test, *rainrate_sequence.shape[1:]))
        # Ground truth: the actual test frames (clean, no overlap)
        ground_truth = test_sequence[: len(predictions)]

        return {
            "model_name": "LINDA-PINN",
            "predictions": predictions,
            "ground_truth": ground_truth,
            "metadata": metadata,
            "training_time": training_time,
            "losses": losses,
            "physics_losses": physics_losses,
            "advection_field": advection_field,
        }

    except Exception as e:
        print(f"PINN training failed: {e}")
        # Return dummy results
        n_pred = min(3, rainrate_sequence.shape[0] - 1)
        dummy_predictions = np.zeros((n_pred, *rainrate_sequence.shape[1:]))
        ground_truth = rainrate_sequence[-n_pred:]

        return {
            "model_name": "LINDA-PINN (Failed)",
            "predictions": dummy_predictions,
            "ground_truth": ground_truth,
            "metadata": metadata,
            "training_time": 0,
            "losses": [],
            "physics_losses": [],
        }


def train_traditional_linda_with_params(
    rainrate_sequence,
    metadata,
    n_input=3,
    n_forecast=6,
    n_ens_members=10,
    vel_pert_p1=1.0,
    vel_pert_p2=0.1,
    vel_pert_p3=0.01,
    vel_pert_p4=0.1,
    vel_pert_p5=0.01,
    kmperpixel=1,
    timestep=5,
):
    """Train traditional LINDA model with custom parameters"""
    print("\n=== Training Traditional LINDA with Custom Parameters ===")

    # Validate inputs
    max_forecast = rainrate_sequence.shape[0] - n_input
    if n_forecast > max_forecast:
        print(f"Warning: n_forecast ({n_forecast}) exceeds available data. Using {max_forecast}")
        n_forecast = max_forecast

    if n_input >= rainrate_sequence.shape[0]:
        raise ValueError(f"n_input ({n_input}) must be less than sequence length ({rainrate_sequence.shape[0]})")

    R_input = rainrate_sequence[:n_input]
    R_truth = rainrate_sequence[n_input : n_input + n_forecast]

    print(f"Using {n_input} input frames to predict {n_forecast} frames")
    print(f"Input shape: {R_input.shape}")
    print(f"Truth shape: {R_truth.shape}")

    # Convert to rain rate
    conv_out = conversion.to_rainrate(R_input, metadata)
    if isinstance(conv_out, tuple) and len(conv_out) >= 1:
        conv_arr = conv_out[0]
    else:
        conv_arr = conv_out

    if isinstance(conv_arr, (list, tuple)):
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr]
        R_input_rr = np.stack(arrs, axis=0)
    elif isinstance(conv_arr, np.ndarray) and conv_arr.dtype == object:
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr.tolist()]
        R_input_rr = np.stack(arrs, axis=0)
    else:
        R_input_rr = np.asarray(conv_arr, dtype=np.float32)

    finite_mask = np.isfinite(R_input_rr)
    if not finite_mask.all():
        R_input_rr[~finite_mask] = 0.0

    # Compute motion field
    motion_field = dense_lucaskanade(R_input_rr)

    # Run LINDA with custom parameters
    linda_forecast = pysteps_linda.forecast(
        R_input_rr,
        motion_field,
        n_forecast,
        kmperpixel=kmperpixel,
        timestep=timestep,
        n_ens_members=n_ens_members,
        vel_pert_kwargs={"p_pert_par": [vel_pert_p1, vel_pert_p2, vel_pert_p3, vel_pert_p4, vel_pert_p5]},
    )

    return {
        "model_name": "Traditional LINDA",
        "predictions": linda_forecast,
        "ground_truth": R_truth,
        "metadata": metadata,
        "motion_field": motion_field,
        "n_input": n_input,
        "n_forecast": n_forecast,
    }


def train_custom_pinn_with_params(
    rainrate_sequence,
    metadata,
    n_input=3,
    n_forecast=3,
    epochs=10,
    learning_rate=0.001,
    weight_decay=1e-5,
    batch_size=1,
    hidden_layers=256,
    num_layers=5,
    initial_sigma=0.0,
    initial_survival=0.8,
    initial_growth=0.1,
):
    """Train custom LINDA-PINN model with custom parameters"""
    print("\n=== Training Custom LINDA-PINN with Custom Parameters ===")

    # Validate inputs - NO overlap between train and test
    min_required = n_input + n_forecast
    if rainrate_sequence.shape[0] < min_required:
        print(f"Warning: Not enough data for n_input={n_input} and n_forecast={n_forecast}")
        n_forecast = max(1, rainrate_sequence.shape[0] - n_input)
        print(f"Adjusted n_forecast to {n_forecast}")

    # Create custom model with specified architecture
    layers = [4] + [hidden_layers] * num_layers + [1]

    # Modify the trainer to accept custom parameters
    trainer = LINDAPINNTrainer()
    trainer.model = LINDAPINNModel(layers=layers)

    # Set initial parameters
    with torch.no_grad():
        trainer.model.log_sigma.fill_(initial_sigma)
        trainer.model.survival_prob.fill_(initial_survival)
        trainer.model.growth_rate.fill_(initial_growth)

    # Update optimizer with custom parameters
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, patience=max(10, epochs // 10))

    # Clean train/test split - NO overlap
    # Test set: last n_forecast frames
    test_start_idx = rainrate_sequence.shape[0] - n_forecast
    test_sequence = rainrate_sequence[test_start_idx:]

    # Training set: all frames BEFORE test set (no overlap)
    train_sequence = rainrate_sequence[:test_start_idx]

    # Ensure we have enough training data for advection computation
    if train_sequence.shape[0] < n_input:
        raise ValueError(f"Not enough training data. Need {n_input} frames, got {train_sequence.shape[0]}")

    # Compute advection field from training data (same as traditional LINDA)
    print("Computing advection field from training data...")
    R_input = train_sequence[:n_input]
    conv_out = conversion.to_rainrate(R_input, metadata)
    if isinstance(conv_out, tuple) and len(conv_out) >= 1:
        conv_arr = conv_out[0]
    else:
        conv_arr = conv_out
    if isinstance(conv_arr, (list, tuple)):
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr]
        R_input_rr = np.stack(arrs, axis=0)
    elif isinstance(conv_arr, np.ndarray) and conv_arr.dtype == object:
        arrs = [np.asarray(a, dtype=np.float32) for a in conv_arr.tolist()]
        R_input_rr = np.stack(arrs, axis=0)
    else:
        R_input_rr = np.asarray(conv_arr, dtype=np.float32)
    finite_mask = np.isfinite(R_input_rr)
    if not finite_mask.all():
        R_input_rr[~finite_mask] = 0.0
    advection_field = dense_lucaskanade(R_input_rr)
    print(f"Advection field computed: shape {advection_field.shape}")

    print(f"Using {n_input} input frames to predict {n_forecast} frames")
    print(f"Training sequence shape: {train_sequence.shape}")
    print(f"Test sequence shape: {test_sequence.shape}")

    try:
        start_time = time.time()
        losses, physics_losses = trainer.train_on_radar_sequence(train_sequence, metadata, epochs=epochs, verbose=True)
        training_time = time.time() - start_time

        # Make predictions using recursive forecasting
        predictions = []

        # Start with the last frame from training data (last known frame before test)
        current_frame = train_sequence[-1]

        for t in range(n_forecast):
            ny, nx = current_frame.shape
            if "xpixelsize" in metadata and "ypixelsize" in metadata:
                x_coords = np.arange(nx) * metadata["xpixelsize"] / 1000.0
                y_coords = np.arange(ny) * metadata["ypixelsize"] / 1000.0
            else:
                x_coords = np.linspace(-100, 100, nx)
                y_coords = np.linspace(-100, 100, ny)

            current_tensor = torch.tensor(current_frame, dtype=torch.float32, device=device)

            # PASS advection field and metadata (critical fix)
            with torch.no_grad():
                next_frame = trainer.model(current_tensor, x_coords, y_coords, float(t), advection_field, metadata)
                next_frame_np = next_frame.cpu().numpy()
                predictions.append(next_frame_np)

            # Use prediction as input for next step (recursive forecasting)
            current_frame = next_frame_np

        predictions = np.array(predictions) if predictions else np.zeros((n_forecast, *rainrate_sequence.shape[1:]))

        # Ground truth: the actual test frames (clean, no overlap with training)
        ground_truth = test_sequence[: len(predictions)]

        return {
            "model_name": "LINDA-PINN",
            "predictions": predictions,
            "ground_truth": ground_truth,
            "metadata": metadata,
            "training_time": training_time,
            "losses": losses,
            "physics_losses": physics_losses,
            "final_params": {
                "sigma": torch.exp(trainer.model.log_sigma).item(),
                "survival": torch.sigmoid(trainer.model.survival_prob).item(),
                "growth": torch.sigmoid(trainer.model.growth_rate).item(),
            },
            "n_input": n_input,
            "n_forecast": n_forecast,
            "advection_field": advection_field,
        }

    except Exception as e:
        print(f"PINN training failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "model_name": "LINDA-PINN (Failed)",
            "predictions": np.zeros((n_forecast, *rainrate_sequence.shape[1:])),
            "ground_truth": rainrate_sequence[-n_forecast:]
            if n_forecast <= rainrate_sequence.shape[0]
            else rainrate_sequence,
            "metadata": metadata,
            "training_time": 0,
            "losses": [],
            "physics_losses": [],
            "n_input": n_input,
            "n_forecast": n_forecast,
        }


def run_comparison(
    # LINDA parameters
    linda_n_input,
    linda_n_forecast,
    linda_n_ens_members,
    linda_vel_p1,
    linda_vel_p2,
    linda_vel_p3,
    linda_vel_p4,
    linda_vel_p5,
    linda_kmperpixel,
    linda_timestep,
    # PINN parameters
    pinn_n_input,
    pinn_n_forecast,
    pinn_epochs,
    pinn_lr,
    pinn_weight_decay,
    pinn_hidden_layers,
    pinn_num_layers,
    pinn_initial_sigma,
    pinn_initial_survival,
    pinn_initial_growth,
    # Data selection
    use_synthetic_data,
):
    """Main function to run the comparison"""

    # Load data
    if use_synthetic_data:
        rainrate_sequence, metadata = generate_synthetic_data()
    else:
        try:
            rainrate_sequence, metadata = load_swiss_radar_data()
        except:
            print("Failed to load real data, using synthetic instead")
            rainrate_sequence, metadata = generate_synthetic_data()

    # Train LINDA
    linda_results = train_traditional_linda_with_params(
        rainrate_sequence,
        metadata,
        n_input=int(linda_n_input),
        n_forecast=int(linda_n_forecast),
        n_ens_members=int(linda_n_ens_members),
        vel_pert_p1=linda_vel_p1,
        vel_pert_p2=linda_vel_p2,
        vel_pert_p3=linda_vel_p3,
        vel_pert_p4=linda_vel_p4,
        vel_pert_p5=linda_vel_p5,
        kmperpixel=linda_kmperpixel,
        timestep=linda_timestep,
    )

    # Train PINN
    pinn_results = train_custom_pinn_with_params(
        rainrate_sequence,
        metadata,
        n_input=int(pinn_n_input),
        n_forecast=int(pinn_n_forecast),
        epochs=int(pinn_epochs),
        learning_rate=pinn_lr,
        weight_decay=pinn_weight_decay,
        hidden_layers=int(pinn_hidden_layers),
        num_layers=int(pinn_num_layers),
        initial_sigma=pinn_initial_sigma,
        initial_survival=pinn_initial_survival,
        initial_growth=pinn_initial_growth,
    )

    # Compute metrics
    linda_metrics = compute_metrics(linda_results["predictions"], linda_results["ground_truth"])
    pinn_metrics = compute_metrics(pinn_results["predictions"], pinn_results["ground_truth"])

    # Create visualizations
    pred_fig = create_prediction_visualization(linda_results, pinn_results)
    loss_fig = create_loss_plot(pinn_results)

    # Format results
    results_text = f"""
    ## Model Comparison Results
    
    ### Traditional LINDA
    - **Input Frames**: {linda_results.get("n_input", "N/A")}
    - **Forecast Frames**: {linda_results.get("n_forecast", "N/A")}
    - **RMSE**: {linda_metrics["rmse"]:.4f}
    - **MAE**: {linda_metrics["mae"]:.4f}
    - **Correlation**: {linda_metrics["correlation"]:.4f}
    - **Accuracy (±20%)**: {linda_metrics["accuracy"]:.2f}%
    
    ### LINDA-PINN
    - **Input Frames**: {pinn_results.get("n_input", "N/A")}
    - **Forecast Frames**: {pinn_results.get("n_forecast", "N/A")}
    - **RMSE**: {pinn_metrics["rmse"]:.4f}
    - **MAE**: {pinn_metrics["mae"]:.4f}
    - **Correlation**: {pinn_metrics["correlation"]:.4f}
    - **Accuracy (±20%)**: {pinn_metrics["accuracy"]:.2f}%
    - **Training Time**: {pinn_results.get("training_time", 0):.2f}s
    
    ### Learned PINN Parameters
    - **Sigma**: {pinn_results.get("final_params", {}).get("sigma", "N/A"):.3f}
    - **Survival**: {pinn_results.get("final_params", {}).get("survival", "N/A"):.3f}
    - **Growth**: {pinn_results.get("final_params", {}).get("growth", "N/A"):.3f}
    
    ### Winner
    - **RMSE**: {"LINDA" if linda_metrics["rmse"] < pinn_metrics["rmse"] else "PINN" if pinn_metrics["rmse"] < linda_metrics["rmse"] else "Tie"}
    - **Accuracy**: {"LINDA" if linda_metrics["accuracy"] > pinn_metrics["accuracy"] else "PINN" if pinn_metrics["accuracy"] > linda_metrics["accuracy"] else "Tie"}
    """

    return results_text, pred_fig, loss_fig
