import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

import pysteps
from pysteps import io, rcparams, motion, datasets
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts import linda as pysteps_linda
from pysteps.utils import conversion
from sklearn.metrics import mean_squared_error
from scipy.ndimage import binary_dilation
import time
from datetime import datetime
import warnings

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import io
from PIL import Image

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LINDAPINNModel(nn.Module):
    def __init__(self, layers=[4, 256, 256, 256, 256, 256, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            nn.init.xavier_uniform_(self.layers[i].weight)

        self.kernel_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.advection_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        for net in [self.kernel_net, self.advection_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

        self.to(device)

        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.survival_prob = nn.Parameter(torch.tensor(0.8))
        self.growth_rate = nn.Parameter(torch.tensor(0.1))
        self.carrying_capacity = nn.Parameter(torch.tensor(10.0))

        # Move model to device
        self.to(device)

    def dispersal_kernel(self, dx, dy, t=None):
        """LINDA redistribution kernel with learnable parameters"""
        sigma = torch.exp(self.log_sigma) + 0.1

        if t is not None:
            if isinstance(t, (int, float)):
                t_tensor = torch.full_like(dx, float(t), device=device)
            else:
                t_tensor = torch.full_like(dx, t.item() if hasattr(t, "item") else float(t), device=device)

            dx_flat = dx.flatten().unsqueeze(1)
            dy_flat = dy.flatten().unsqueeze(1)
            t_flat = t_tensor.flatten().unsqueeze(1)

            kernel_input = torch.cat([dx_flat, dy_flat, t_flat], dim=1)
            kernel_weight = torch.sigmoid(self.kernel_net(kernel_input))
            kernel_weight = kernel_weight.reshape(dx.shape)
        else:
            kernel_weight = torch.tensor(1.0, device=device)

        kernel = kernel_weight * torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        kernel = kernel / (2 * np.pi * sigma**2)

        return kernel

    def compute_integral_term(self, R_field, x_coords, y_coords, t, metadata=None):
        """Compute the integral term in LINDA equation using FFT-based convolution

        The kernel is normalized to preserve mass: ∫K(x-y)dy = 1
        """
        ny, nx = R_field.shape

        # Get pixel scales from metadata if available
        if metadata and "xpixelsize" in metadata and "ypixelsize" in metadata:
            dx_scale = metadata["xpixelsize"] / 1000.0  # Convert to km
            dy_scale = metadata["ypixelsize"] / 1000.0
        elif len(x_coords) > 1 and len(y_coords) > 1:
            dx_scale = x_coords[1] - x_coords[0]
            dy_scale = y_coords[1] - y_coords[0]
        else:
            dx_scale = 1.0
            dy_scale = 1.0

        # Create coordinate grids for kernel - centered at origin
        Y_grid, X_grid = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32, device=device) - ny // 2,
            torch.arange(nx, dtype=torch.float32, device=device) - nx // 2,
            indexing="ij",
        )

        # Scale coordinates
        X_grid = X_grid * dx_scale
        Y_grid = Y_grid * dy_scale

        # Compute dispersal kernel centered at origin
        kernel = self.dispersal_kernel(X_grid, Y_grid, t)

        # Normalize kernel to ensure mass conservation: sum(kernel) * pixel_area = 1
        pixel_area = dx_scale * dy_scale
        kernel_sum = torch.sum(kernel) * pixel_area
        kernel = kernel / kernel_sum

        # Apply fftshift to move zero frequency to center
        kernel_shifted = torch.fft.fftshift(kernel)

        # Compute FFT of both kernel and field
        kernel_fft = torch.fft.rfft2(kernel_shifted)
        field_fft = torch.fft.rfft2(R_field)

        # Multiply in frequency domain (convolution theorem)
        convolved_fft = kernel_fft * field_fft

        # Inverse FFT to get result
        integral_result = torch.fft.irfft2(convolved_fft, s=(ny, nx))

        # Ensure result is real and positive
        integral_result = torch.real(integral_result)
        integral_result = torch.clamp(integral_result, min=0.0)

        return integral_result

    def apply_advection(self, field, advection_field, metadata):
        """Apply semi-Lagrangian advection to field (differentiable)

        Args:
            field: 2D tensor (ny, nx) - the field to advect
            advection_field: 3D numpy array (2, ny, nx) - velocity field [u, v]
            metadata: dict with pixel sizes
        """
        # Ensure field is on device
        if isinstance(field, np.ndarray):
            field = torch.tensor(field, dtype=torch.float32, device=device)

        # Get velocity components
        u = advection_field[0]  # x-component
        v = advection_field[1]  # y-component

        # Get grid dimensions
        ny, nx = field.shape

        # Time step (5 minutes in seconds)
        dt = 5 * 60

        # Pixel sizes in meters
        dx = metadata.get("xpixelsize", 1000)
        dy = metadata.get("ypixelsize", 1000)

        # Convert velocities from pixels/timestep to grid units
        u_grid = torch.tensor(u * dt / dx, dtype=torch.float32, device=device)
        v_grid = torch.tensor(v * dt / dy, dtype=torch.float32, device=device)

        # Create coordinate grids (normalized to [-1, 1] for grid_sample)
        y_coords = torch.arange(ny, dtype=torch.float32, device=device)
        x_coords = torch.arange(nx, dtype=torch.float32, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Backward trajectories
        X_back = X - u_grid
        Y_back = Y - v_grid

        # Normalize to [-1, 1] for grid_sample
        X_norm = 2.0 * X_back / (nx - 1) - 1.0
        Y_norm = 2.0 * Y_back / (ny - 1) - 1.0

        # Clip to [-1, 1] range
        X_norm = torch.clamp(X_norm, -1, 1)
        Y_norm = torch.clamp(Y_norm, -1, 1)

        # Stack and reshape for grid_sample (1, ny, nx, 2)
        grid = torch.stack([X_norm, Y_norm], dim=-1).unsqueeze(0)

        # Reshape field for grid_sample (1, 1, ny, nx)
        field_reshaped = field.unsqueeze(0).unsqueeze(0)

        # Apply differentiable bilinear interpolation
        advected = torch.nn.functional.grid_sample(
            field_reshaped, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # Reshape back to (ny, nx)
        advected = advected.squeeze(0).squeeze(0)

        return advected

    def linda_equation(self, R_current, x_coords, y_coords, t, advection_field, metadata):
        """Implement the actual LINDA integro-difference equation

        LINDA equation:
        R(x,t+1) = s * ∫∫ K(x-y) * R(y,t) dy + g * R(x,t) * (1 - R(x,t)/C)

        With advection applied via operator splitting:
        R_adv(x,t) = R(x - v*dt, t)  (semi-Lagrangian backtracking)
        Final: R(x,t+1) = s * ∫∫ K(x-y) * R_adv(y,t) dy + growth_term
        """
        # Ensure R_current is on device
        if not R_current.is_cuda and device.type == "cuda":
            R_current = R_current.to(device)

        # 1. Apply advection first (operator splitting - advection step)
        if advection_field is not None:
            R_advected = self.apply_advection(R_current, advection_field, metadata)

            # Use advection_net to learn modulation factor from velocity statistics
            u_mean = float(np.mean(advection_field[0]))
            v_mean = float(np.mean(advection_field[1]))
            u_std = float(np.std(advection_field[0]))
            v_std = float(np.std(advection_field[1]))

            advection_input = torch.tensor([u_mean, v_mean, u_std, v_std], dtype=torch.float32, device=device)
            advection_modulation = self.advection_net(advection_input)
        else:
            R_advected = R_current
            advection_modulation = torch.tensor(1.0, device=device)

        # 2. Dispersal term (integral) - applied to advected field
        integral_term = self.compute_integral_term(R_advected, x_coords, y_coords, t, metadata)
        survival_prob = torch.sigmoid(self.survival_prob)
        dispersal_term = survival_prob * integral_term * advection_modulation

        # 3. Growth term (logistic growth)
        growth_rate = torch.sigmoid(self.growth_rate)
        carrying_capacity = F.softplus(self.carrying_capacity) + 1.0
        growth_term = growth_rate * R_advected * (1 - R_advected / carrying_capacity)

        # Combine terms according to LINDA integro-difference structure
        R_next = dispersal_term + growth_term

        return torch.clamp(R_next, min=0.0)

    def forward(self, R_field, x_coords, y_coords, t, advection_field=None, metadata=None):
        """Forward pass with advection

        Args:
            R_field: 2D tensor (ny, nx) - current rainrate field
            x_coords: 1D array - x coordinates in km
            y_coords: 1D array - y coordinates in km
            t: float - time step
            advection_field: 3D numpy array (2, ny, nx) - velocity field
            metadata: dict - contains xpixelsize, ypixelsize, etc.
        """
        if not R_field.is_cuda and device.type == "cuda":
            R_field = R_field.to(device)

        # Use provided metadata or create minimal default
        if metadata is None:
            metadata = {}

        return self.linda_equation(R_field, x_coords, y_coords, t, advection_field, metadata)


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


def load_swiss_radar_data():
    """Load Swiss radar data from pysteps"""
    try:
        # Try to use built-in datasets first
        print("Attempting to download pysteps data...")
        root_path = pysteps.datasets.download_pysteps_data()

        # Use sample date from dataset
        date = datetime.strptime("201609080000", "%Y%m%d%H%M")
        data_source = "mch"

        # Create file list
        fns = pysteps.datasets.create_file_list(root_path, "mchrzc12", "201609080000", "201609081200", timestep=5)

        if len(fns) == 0:
            raise FileNotFoundError("No files found in dataset")

        print(f"Found {len(fns)} radar files")

        # Get importer
        importer = io.get_method("mchrzc12")

        # Read the data
        rainrate_sequence, _, metadata = io.read_timeseries(
            fns, importer, **importer.kwargs if hasattr(importer, "kwargs") else {}
        )

        print(f"Loaded radar sequence shape: {rainrate_sequence.shape}")
        print(f"Pixel resolution: {metadata.get('xpixelsize', 'unknown')}m x {metadata.get('ypixelsize', 'unknown')}m")

        return rainrate_sequence, metadata

    except Exception as e:
        print(f"Failed to load pysteps data: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic radar data for testing"""
    np.random.seed(42)
    nt, ny, nx = 12, 256, 256

    # Create synthetic precipitation patterns
    rainrate_sequence = np.zeros((nt, ny, nx))

    for t in range(nt):
        # Moving rain cells
        center_x = int(nx * 0.3 + (nx * 0.4) * t / nt)
        center_y = int(ny * 0.5 + 20 * np.sin(t * 0.5))

        # Create Gaussian rain cell
        y_grid, x_grid = np.mgrid[0:ny, 0:nx]
        rain_cell = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * 30**2))

        # Add some noise and evolution
        evolution = 1.0 + 0.2 * np.sin(t * 0.3)
        rainrate_sequence[t] = evolution * rain_cell * (5 + 2 * np.random.random())

        # Add smaller cells
        for i in range(2):
            small_x = int(np.random.random() * nx)
            small_y = int(np.random.random() * ny)
            small_cell = np.exp(-((x_grid - small_x) ** 2 + (y_grid - small_y) ** 2) / (2 * 15**2))
            rainrate_sequence[t] += 0.5 * small_cell * np.random.random()

    # Create basic metadata
    metadata = {
        "xpixelsize": 1000.0,  # 1km resolution
        "ypixelsize": 1000.0,
        "unit": "mm/h",
        "accutime": 5.0,  # 5 minute accumulation
        "transform": None,
    }

    print(f"Generated synthetic data shape: {rainrate_sequence.shape}")
    return rainrate_sequence, metadata


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


def create_gradio_app():
    with gr.Blocks(title="LINDA vs LINDA-PINN Comparison") as app:
        gr.Markdown("""
        # LINDA vs LINDA-PINN Weather Nowcasting Comparison
        
        Compare traditional LINDA with Physics-Informed Neural Network (PINN) approach for precipitation nowcasting.
        Adjust hyperparameters for both models and see how they perform!
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### LINDA Parameters")
                gr.Markdown("#### Data Configuration")
                linda_n_input = gr.Slider(3, 10, value=3, step=1, label="Input Frames (n_input)")
                linda_n_forecast = gr.Slider(1, 256, value=6, step=1, label="Forecast Frames (n_forecast)")

                gr.Markdown("#### Model Parameters")
                linda_n_ens = gr.Slider(1, 50, value=10, step=1, label="Ensemble Members")
                linda_vel_p1 = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Velocity Perturbation P1")
                linda_vel_p2 = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Velocity Perturbation P2")
                linda_vel_p3 = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Velocity Perturbation P3")
                linda_vel_p4 = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Velocity Perturbation P4")
                linda_vel_p5 = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Velocity Perturbation P5")
                linda_km = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="KM per Pixel")
                linda_timestep = gr.Slider(1, 15, value=5, step=1, label="Timestep (minutes)")

            with gr.Column():
                gr.Markdown("### PINN Parameters")
                gr.Markdown("#### Data Configuration")
                pinn_n_input = gr.Slider(3, 10, value=3, step=1, label="Input Frames (n_input)")
                pinn_n_forecast = gr.Slider(1, 256, value=3, step=1, label="Forecast Frames (n_forecast)")

                gr.Markdown("#### Model Parameters")
                pinn_epochs = gr.Slider(5, 100, value=10, step=5, label="Training Epochs")
                pinn_lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")
                pinn_weight_decay = gr.Slider(1e-6, 1e-3, value=1e-5, step=1e-6, label="Weight Decay")
                pinn_hidden = gr.Slider(64, 512, value=256, step=64, label="Hidden Layer Size")
                pinn_layers = gr.Slider(2, 8, value=5, step=1, label="Number of Layers")
                pinn_sigma = gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="Initial Log Sigma")
                pinn_survival = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Initial Survival Probability")
                pinn_growth = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Initial Growth Rate")

        with gr.Row():
            use_synthetic = gr.Checkbox(value=True, label="Use Synthetic Data (faster)")
            run_btn = gr.Button("Run Comparison", variant="primary")

        with gr.Row():
            results_output = gr.Markdown()

        with gr.Row():
            predictions_plot = gr.Plot(label="Predictions Comparison")
            loss_plot = gr.Plot(label="PINN Training Loss")

        run_btn.click(
            fn=run_comparison,
            inputs=[
                linda_n_input,
                linda_n_forecast,
                linda_n_ens,
                linda_vel_p1,
                linda_vel_p2,
                linda_vel_p3,
                linda_vel_p4,
                linda_vel_p5,
                linda_km,
                linda_timestep,
                pinn_n_input,
                pinn_n_forecast,
                pinn_epochs,
                pinn_lr,
                pinn_weight_decay,
                pinn_hidden,
                pinn_layers,
                pinn_sigma,
                pinn_survival,
                pinn_growth,
                use_synthetic,
            ],
            outputs=[results_output, predictions_plot, loss_plot],
        )

        gr.Markdown("""
        ### About
        - **LINDA**: Lagrangian Integro-Difference equation with Nowcasting and Data Assimilation
        - **PINN/LINDA-PINN**: LINDA-inspired integro-difference PINN model
        - **n_input**: Number of past frames used to make predictions
        - **n_forecast**: Number of future frames to predict
        - Metrics shown are computed on test data
        """)

    return app


# Launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)
