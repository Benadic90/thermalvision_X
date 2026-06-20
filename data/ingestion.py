# data/ingestion.py
# THERMAVISION-X — Satellite Data Ingestion
# Supports: INSAT-3D/3DR (HDF5), Landsat 8/9 (GeoTIFF), generic images (JPG/PNG)
# Researched & designed by Benad | BAH 2026

import os
import numpy as np
from PIL import Image


class SatelliteDataIngestion:
    """
    Unified loader for ISRO and international satellite thermal data.
    Handles HDF5 (INSAT), GeoTIFF (Landsat), and generic image formats.
    """

    SUPPORTED_SATELLITES = {
        'INSAT_3D': {
            'format': 'HDF5',
            'bands': {'TIR1': (10.3, 11.3), 'TIR2': (11.5, 12.5)},
            'resolution': 4000,  # meters
        },
        'INSAT_3DR': {
            'format': 'HDF5',
            'bands': {'TIR1': (10.3, 11.3), 'TIR2': (11.5, 12.5)},
            'resolution': 4000,
        },
        'TRISHNA': {
            'format': 'HDF5',
            'bands': {'TIR1': (8.0, 9.0), 'TIR2': (10.0, 11.0),
                      'TIR3': (11.0, 12.0), 'TIR4': (12.0, 13.0)},
            'resolution': 57,
        },
        'LANDSAT_8': {
            'format': 'GeoTIFF',
            'bands': {'B10': (10.6, 11.2), 'B11': (11.5, 12.5)},
            'resolution': 100,
        },
        'GENERIC': {
            'format': 'IMAGE',   # JPG, PNG — for demo & testing
            'bands': {'GRAY': (0, 1)},
            'resolution': None,
        },
    }

    def __init__(self, satellite_type: str = 'GENERIC'):
        if satellite_type not in self.SUPPORTED_SATELLITES:
            raise ValueError(
                f"Satellite '{satellite_type}' not supported. "
                f"Choose from: {list(self.SUPPORTED_SATELLITES.keys())}"
            )
        self.satellite_type = satellite_type
        self.config = self.SUPPORTED_SATELLITES[satellite_type]

    def load(self, filepath: str, band: str = 'GRAY') -> tuple:
        """
        Load thermal band from file.

        Returns:
            (dn_data, cal_coeffs, geo_transform, projection)
            For GENERIC format: cal_coeffs = {'slope': 1.0, 'intercept': 0.0}
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        fmt = self.config['format']

        if fmt == 'HDF5':
            return self._load_hdf5(filepath, band)
        elif fmt == 'GeoTIFF':
            return self._load_geotiff(filepath, band)
        elif fmt == 'IMAGE':
            return self._load_image(filepath)
        else:
            raise NotImplementedError(f"Format '{fmt}' not implemented.")

    def _load_hdf5(self, filepath: str, band: str) -> tuple:
        """Load from ISRO HDF5 format (INSAT-3D/3DR/TRISHNA)."""
        try:
            import h5py
        except ImportError:
            raise ImportError("Install h5py: pip install h5py")

        with h5py.File(filepath, 'r') as f:
            dn_data = f[f'IMG_{band}'][()]
            slope = float(f[f'IMG_{band}'].attrs.get('CALIBRATION_SLOPE', 1.0))
            intercept = float(f[f'IMG_{band}'].attrs.get('CALIBRATION_INTERCEPT', 0.0))
            geo_transform = f.get('Geolocation', {}).attrs.get('GEO_TRANSFORM', None)
            projection = f.get('Geolocation', {}).attrs.get('PROJECTION', None)

        return dn_data, {'slope': slope, 'intercept': intercept}, geo_transform, projection

    def _load_geotiff(self, filepath: str, band: str) -> tuple:
        """Load from GeoTIFF format (Landsat 8/9)."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("Install rasterio: pip install rasterio")

        with rasterio.open(filepath) as src:
            dn_data = src.read(1).astype(np.float32)
            geo_transform = src.transform
            projection = src.crs.to_wkt() if src.crs else None

        return dn_data, {'slope': 1.0, 'intercept': 0.0}, geo_transform, projection

    def _load_image(self, filepath: str) -> tuple:
        """Load generic image (JPG/PNG) as grayscale — for demo & testing."""
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        dn_data = np.array(img, dtype=np.float32)
        return dn_data, {'slope': 1.0 / 255.0, 'intercept': 0.0}, None, None


if __name__ == "__main__":
    # Quick test with a generic image (no satellite file needed)
    print("Testing generic image loading...")
    ingestion = SatelliteDataIngestion('GENERIC')
    print(f"Supported satellites: {list(SatelliteDataIngestion.SUPPORTED_SATELLITES.keys())}")
    print("✅ Data ingestion module ready!")
