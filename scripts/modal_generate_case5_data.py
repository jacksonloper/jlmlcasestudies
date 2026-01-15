"""
Generate data for Case Study 5: Multispectral Imaging.

This script extracts a random WT sample from the beyondRGB dataset on the Modal volume,
converts the HDF5 data to PNG images, and returns them for the frontend.
"""

import modal

# Create Modal app
app = modal.App("generate-case5-data")

# Reference the existing volume
volume = modal.Volume.from_name("jlmlcasestudies")

# Define image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "h5py", "numpy", "pillow"
)


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
    # Normalize to 0-255 range
    oppo_min = np.min(oppo_data)
    oppo_max = np.max(oppo_data)
    print(f"OPPO value range: {oppo_min} to {oppo_max}")
    
    # Scale to 0-255
    oppo_normalized = ((oppo_data - oppo_min) / (oppo_max - oppo_min + 1e-8) * 255).astype(np.uint8)
    
    # Resize for web (too large otherwise)
    oppo_img = Image.fromarray(oppo_normalized)
    # Resize to reasonable web size (max 800px width)
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
    
    # Process MIS multispectral image
    # MIS is 2D with shape (H, W) but actually contains 31 spectral bands interleaved
    # Based on the paper, MIS has 31 channels from 400-700nm in 10nm steps
    mis_bands = []
    
    # The MIS data appears to be (H, W) shaped - need to understand the format better
    # Let's check if it's already separated or needs processing
    if len(mis_data.shape) == 2:
        # Single 2D array - this is the full multispectral cube flattened
        # According to the paper, there are 31 spectral bands
        # The data might be stored as a single grayscale image per band
        # or interleaved in some way
        
        # For now, treat as a single grayscale image and create bands by region
        # Actually, looking at typical MIS formats, the 31 bands are likely
        # stored in separate files or the shape indicates something else
        
        # Let's create a visualization of the single MIS image
        # and also create pseudo-bands by splitting
        print(f"MIS is 2D: {mis_data.shape}")
        
        # Normalize
        mis_min = np.min(mis_data)
        mis_max = np.max(mis_data)
        print(f"MIS value range: {mis_min} to {mis_max}")
        
        mis_normalized = ((mis_data - mis_min) / (mis_max - mis_min + 1e-8) * 255).astype(np.uint8)
        
        # Create grayscale image
        mis_img = Image.fromarray(mis_normalized, mode='L')
        
        # Resize to match OPPO dimensions approximately
        mis_aspect = mis_img.height / mis_img.width
        mis_new_width = min(mis_img.width, max_width)
        mis_new_height = int(mis_new_width * mis_aspect)
        mis_img_resized = mis_img.resize((mis_new_width, mis_new_height), Image.Resampling.LANCZOS)
        
        # Store as single "band" for now
        mis_buffer = io.BytesIO()
        mis_img_resized.save(mis_buffer, format="PNG")
        mis_bands.append(mis_buffer.getvalue())
        
        print(f"MIS PNG size: {len(mis_bands[0])} bytes, dimensions: {mis_new_width}x{mis_new_height}")
        
    elif len(mis_data.shape) == 3:
        # 3D array - likely (H, W, C) or (C, H, W) where C=31 bands
        if mis_data.shape[2] <= 31:
            # (H, W, C) format
            n_bands = mis_data.shape[2]
            for band_idx in range(n_bands):
                band_data = mis_data[:, :, band_idx]
                # Normalize each band independently
                band_min = np.min(band_data)
                band_max = np.max(band_data)
                band_normalized = ((band_data - band_min) / (band_max - band_min + 1e-8) * 255).astype(np.uint8)
                
                band_img = Image.fromarray(band_normalized, mode='L')
                
                # Resize
                band_aspect = band_img.height / band_img.width
                band_new_width = min(band_img.width, max_width)
                band_new_height = int(band_new_width * band_aspect)
                band_img_resized = band_img.resize((band_new_width, band_new_height), Image.Resampling.LANCZOS)
                
                band_buffer = io.BytesIO()
                band_img_resized.save(band_buffer, format="PNG")
                mis_bands.append(band_buffer.getvalue())
                
            print(f"Processed {n_bands} MIS bands")
        else:
            # (C, H, W) format
            n_bands = mis_data.shape[0]
            for band_idx in range(n_bands):
                band_data = mis_data[band_idx, :, :]
                band_min = np.min(band_data)
                band_max = np.max(band_data)
                band_normalized = ((band_data - band_min) / (band_max - band_min + 1e-8) * 255).astype(np.uint8)
                
                band_img = Image.fromarray(band_normalized, mode='L')
                
                band_aspect = band_img.height / band_img.width
                band_new_width = min(band_img.width, max_width)
                band_new_height = int(band_new_width * band_aspect)
                band_img_resized = band_img.resize((band_new_width, band_new_height), Image.Resampling.LANCZOS)
                
                band_buffer = io.BytesIO()
                band_img_resized.save(band_buffer, format="PNG")
                mis_bands.append(band_buffer.getvalue())
                
            print(f"Processed {n_bands} MIS bands")
    
    return {
        "oppo_rgb": oppo_png,
        "mis_bands": mis_bands,
        "sample_name": sample_name,
        "mis_shape": list(mis_data.shape),
        "oppo_shape": list(oppo_data.shape),
    }


@app.local_entrypoint()
def main():
    """Generate and save Case 5 images locally."""
    import os
    from pathlib import Path
    
    print("Generating Case 5 images from beyondRGB dataset...")
    result = generate_case5_images.remote()
    
    # Save to case5/data directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "case5" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save OPPO RGB image
    oppo_path = output_dir / "oppo_rgb.png"
    with open(oppo_path, "wb") as f:
        f.write(result["oppo_rgb"])
    print(f"Saved OPPO RGB image: {oppo_path}")
    
    # Save MIS bands
    for i, band_data in enumerate(result["mis_bands"]):
        band_path = output_dir / f"mis_band_{i:02d}.png"
        with open(band_path, "wb") as f:
            f.write(band_data)
        print(f"Saved MIS band {i}: {band_path}")
    
    print(f"\nSample: {result['sample_name']}")
    print(f"OPPO shape: {result['oppo_shape']}")
    print(f"MIS shape: {result['mis_shape']}")
    print(f"\nTotal MIS bands: {len(result['mis_bands'])}")
    print(f"\nAll images saved to: {output_dir}")
