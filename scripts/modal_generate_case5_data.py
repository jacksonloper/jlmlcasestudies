"""
Generate data for Case Study 5: Multispectral Imaging.

This script extracts a random WT sample from the beyondRGB dataset on the Modal volume,
converts the HDF5 data to PNG images, and returns them for the frontend.

The MIS camera uses a 6x6 mosaic filter array where different pixel locations correspond
to different spectral bands. We extract all 31 bands (400-700nm at 10nm intervals) from
the mosaic pattern.
"""

import modal

# Create Modal app
app = modal.App("generate-case5-data")

# Reference the existing volume
volume = modal.Volume.from_name("jlmlcasestudies")

# Define image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "h5py", "numpy", "pillow", "scipy"
)


# MIS mosaic pattern - 6x6 pattern with 31 spectral bands + 5 reference pixels
# Band indices 0-30 correspond to wavelengths 400-700nm at 10nm intervals
# The mosaic pattern maps (row % 6, col % 6) -> band index
# This is based on the Beyond RGB paper's mosaic filter array design
MIS_MOSAIC_PATTERN = [
    [0, 1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35],  # 31-35 are reference/unused
]


def extract_spectral_bands(mosaic_data, n_bands=31):
    """
    Extract spectral bands from a mosaic image.
    
    The MIS camera captures 31 spectral bands in a 6x6 mosaic pattern.
    Each band's pixels are at specific (row % 6, col % 6) positions.
    
    We use bilinear interpolation to create full-resolution images for each band.
    """
    import numpy as np
    from scipy.ndimage import zoom
    
    h, w = mosaic_data.shape
    bands = []
    
    # Calculate output dimensions (1/6 of original, then upscaled)
    out_h = h // 6
    out_w = w // 6
    
    for band_idx in range(n_bands):
        # Find position in 6x6 pattern for this band
        band_row = band_idx // 6
        band_col = band_idx % 6
        
        # Extract pixels for this band (every 6th pixel starting at pattern position)
        band_data = mosaic_data[band_row::6, band_col::6]
        
        # Resize to match a reasonable output resolution using bilinear interpolation
        # Scale up by 2x to get a smoother image
        scale_factor = 2.0
        band_upscaled = zoom(band_data, scale_factor, order=1)
        
        bands.append(band_upscaled)
    
    return bands


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=10 * 60,
)
def generate_case5_images():
    """
    Extract a random WT sample and convert to PNG images.
    
    Returns:
        Dictionary with:
        - oppo_rgb: PNG bytes for OPPO RGB image
        - mis_bands: List of 31 PNG bytes, one per spectral band
        - sample_name: Name of the selected sample
    """
    import os
    import random
    import h5py
    import numpy as np
    from PIL import Image
    import io
    
    base_path = "/data/beyondRGB/beyondRGB/clb/BLUE_blue"
    
    # Get list of available lighting conditions with WT data
    wt_dirs = []
    for lighting in os.listdir(base_path):
        wt_path = os.path.join(base_path, lighting, "WT")
        if os.path.isdir(wt_path):
            # Check if both oppo.h5 and MIS.h5 exist
            if os.path.exists(os.path.join(wt_path, "oppo.h5")) and \
               os.path.exists(os.path.join(wt_path, "MIS.h5")):
                wt_dirs.append((lighting, wt_path))
    
    print(f"Found {len(wt_dirs)} WT directories with both oppo.h5 and MIS.h5")
    
    # Select a random sample (using seed for reproducibility)
    random.seed(42)
    selected = random.choice(wt_dirs)
    sample_name = selected[0]
    wt_path = selected[1]
    
    print(f"Selected sample: {sample_name}")
    
    # Load OPPO RGB image
    oppo_path = os.path.join(wt_path, "oppo.h5")
    print(f"Loading OPPO image from {oppo_path}")
    with h5py.File(oppo_path, "r") as f:
        oppo_data = f["oppo"][:]
        print(f"OPPO shape: {oppo_data.shape}, dtype: {oppo_data.dtype}")
    
    # Load MIS multispectral image
    mis_path = os.path.join(wt_path, "MIS.h5")
    print(f"Loading MIS image from {mis_path}")
    with h5py.File(mis_path, "r") as f:
        mis_data = f["MIS"][:]
        print(f"MIS shape: {mis_data.shape}, dtype: {mis_data.dtype}")
    
    # Process OPPO RGB image
    oppo_min = np.min(oppo_data)
    oppo_max = np.max(oppo_data)
    print(f"OPPO value range: {oppo_min} to {oppo_max}")
    
    # Scale to 0-255
    oppo_normalized = ((oppo_data - oppo_min) / (oppo_max - oppo_min + 1e-8) * 255).astype(np.uint8)
    
    # Resize for web (max 800px width)
    oppo_img = Image.fromarray(oppo_normalized)
    max_width = 800
    aspect_ratio = oppo_img.height / oppo_img.width
    new_width = min(oppo_img.width, max_width)
    new_height = int(new_width * aspect_ratio)
    oppo_img_resized = oppo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to PNG bytes
    oppo_buffer = io.BytesIO()
    oppo_img_resized.save(oppo_buffer, format="PNG")
    oppo_png = oppo_buffer.getvalue()
    
    print(f"OPPO PNG size: {len(oppo_png)} bytes, dimensions: {new_width}x{new_height}")
    
    # Process MIS multispectral image - extract all 31 bands
    print(f"Extracting 31 spectral bands from MIS mosaic (shape: {mis_data.shape})")
    
    mis_bands = []
    
    if len(mis_data.shape) == 2:
        # 2D mosaic - extract 31 bands using the mosaic pattern
        spectral_bands = extract_spectral_bands(mis_data, n_bands=31)
        
        for band_idx, band_data in enumerate(spectral_bands):
            # Normalize each band independently
            band_min = np.min(band_data)
            band_max = np.max(band_data)
            band_normalized = ((band_data - band_min) / (band_max - band_min + 1e-8) * 255).astype(np.uint8)
            
            # Create grayscale image
            band_img = Image.fromarray(band_normalized, mode='L')
            
            # Resize to match OPPO dimensions
            band_img_resized = band_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PNG bytes
            band_buffer = io.BytesIO()
            band_img_resized.save(band_buffer, format="PNG")
            mis_bands.append(band_buffer.getvalue())
            
            wavelength = 400 + band_idx * 10
            if band_idx % 10 == 0:
                print(f"  Band {band_idx}: {wavelength}nm - {len(mis_bands[-1])} bytes")
        
        print(f"Extracted {len(mis_bands)} spectral bands")
        
    elif len(mis_data.shape) == 3:
        # 3D array - already separated bands
        if mis_data.shape[2] <= 31:
            n_bands = mis_data.shape[2]
        else:
            n_bands = mis_data.shape[0]
            mis_data = np.transpose(mis_data, (1, 2, 0))
            
        for band_idx in range(min(n_bands, 31)):
            band_data = mis_data[:, :, band_idx]
            band_min = np.min(band_data)
            band_max = np.max(band_data)
            band_normalized = ((band_data - band_min) / (band_max - band_min + 1e-8) * 255).astype(np.uint8)
            
            band_img = Image.fromarray(band_normalized, mode='L')
            band_img_resized = band_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            band_buffer = io.BytesIO()
            band_img_resized.save(band_buffer, format="PNG")
            mis_bands.append(band_buffer.getvalue())
        
        print(f"Processed {len(mis_bands)} MIS bands")
    
    return {
        "oppo_rgb": oppo_png,
        "mis_bands": mis_bands,
        "sample_name": sample_name,
        "mis_shape": list(mis_data.shape),
        "oppo_shape": list(oppo_data.shape),
        "n_bands": len(mis_bands),
    }


@app.local_entrypoint()
def main():
    """Generate and save Case 5 images locally."""
    import os
    from pathlib import Path
    
    print("Generating Case 5 images from beyondRGB dataset...")
    result = generate_case5_images.remote()
    
    # Save to public/case5/data directory for frontend
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "public" / "case5" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save OPPO RGB image
    oppo_path = output_dir / "oppo_rgb.png"
    with open(oppo_path, "wb") as f:
        f.write(result["oppo_rgb"])
    print(f"Saved OPPO RGB image: {oppo_path}")
    
    # Save all MIS bands
    for i, band_data in enumerate(result["mis_bands"]):
        band_path = output_dir / f"mis_band_{i:02d}.png"
        with open(band_path, "wb") as f:
            f.write(band_data)
        wavelength = 400 + i * 10
        print(f"Saved MIS band {i} ({wavelength}nm): {band_path}")
    
    print(f"\nSample: {result['sample_name']}")
    print(f"OPPO shape: {result['oppo_shape']}")
    print(f"MIS shape: {result['mis_shape']}")
    print(f"\nTotal MIS bands: {result['n_bands']}")
    print(f"\nAll images saved to: {output_dir}")
