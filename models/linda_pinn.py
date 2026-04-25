import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
