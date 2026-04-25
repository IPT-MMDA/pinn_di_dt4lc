from datetime import datetime

import numpy as np
import pysteps
from pysteps import io


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
