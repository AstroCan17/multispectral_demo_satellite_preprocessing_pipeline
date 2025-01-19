# Multispectral Demo Satellite Preprocessing Pipeline

A demonstration preprocessing pipeline for pushbroom multispectral optical instruments (under development).

## Overview

This project implements an image processing pipeline for multispectral satellite imagery, with features including:

- Level 0 data processing and decoding
- Level 1 radiometric corrections including:
    - Non-uniformity correction (NUC)
    - Dark current correction
    - Denoising using various filters
    - Image sharpening and deconvolution
- Band co-registration
- Georeferencing
- Pansharpening

## Requirements

- Python 3.x
- Required packages:
    - NumPy
    - OpenCV
    - GDAL
    - rasterio
    - scikit-image
    - matplotlib
    - Earth Engine API

## Usage

1. Clone this repository
2. Install required dependencies
3. Run the main pipeline:
```python
python 02_scripts/main_ips_v6.py
```

## Project Structure

- `02_scripts`
    - Core processing scripts
        - `level_0.py` - Level 0 processing
        - `level_1.py` - Level 1 processing and corrections  
        - `band_coreg.py` - Band co-registration
        - `georeferencing_v1.py` - Georeferencing
        - `metrics_ips.py` - Quality metrics
        - `pansharp.py` - Pansharpening

## Coming Soon

- Unit tests
- On-orbit MTF estimation
- PSF sharpening
- Additional pansharpening methods
- Atmospheric correction

## License

This project is licensed under the GNU GPL v3 - see the `LICENSE` file for details.
