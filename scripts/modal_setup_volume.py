"""
Modal script to create the jlmlcasestudies volume and download beyondRGB.zip.005 from Zenodo.

This script:
1. Creates a Modal volume named 'jlmlcasestudies' (or uses existing one)
2. Downloads beyondRGB.zip.005 from Zenodo to a 'beyondRGB' subdirectory

Note: beyondRGB.zip.005 is part of a multi-part zip archive, so it's stored as-is.
"""

import modal

# Constants
ZENODO_URL = "https://zenodo.org/records/16848482/files/beyondRGB.zip.005?download=1"
DOWNLOAD_TIMEOUT = 1800  # 30 minutes timeout for download

# Create Modal app
app = modal.App("setup-jlmlcasestudies-volume")

# Create or get the volume
volume = modal.Volume.from_name("jlmlcasestudies", create_if_missing=True)

# Define the image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install("curl")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=30 * 60,  # 30 minute timeout for download
)
def download_file():
    """
    Download beyondRGB.zip.005 from Zenodo to the volume.
    The file is stored as-is since it's part of a multi-part zip archive.
    """
    import subprocess
    import os

    # Create the target directory
    target_dir = "/vol/beyondRGB"
    os.makedirs(target_dir, exist_ok=True)

    # Download URL
    output_file = f"{target_dir}/beyondRGB.zip.005"

    print(f"Downloading from {ZENODO_URL}...")
    result = subprocess.run(
        ["curl", "-L", "-o", output_file, ZENODO_URL],
        capture_output=True,
        text=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    if result.returncode != 0:
        print(f"Download failed: stderr={result.stderr}, stdout={result.stdout}")
        raise RuntimeError(f"Download failed: stderr={result.stderr}, stdout={result.stdout}")

    # Check file size
    file_size = os.path.getsize(output_file)
    print(f"Downloaded {file_size} bytes to {output_file}")

    # List files in target directory
    print(f"\nFiles in {target_dir}:")
    for f in os.listdir(target_dir):
        full_path = os.path.join(target_dir, f)
        size = os.path.getsize(full_path)
        print(f"  {f} ({size} bytes)")

    # Commit the volume changes
    volume.commit()
    print("\nVolume committed successfully")

    return f"Success: Downloaded {file_size} bytes to {output_file}"


@app.local_entrypoint()
def main():
    """
    Main entrypoint for running the volume setup on Modal.
    """
    print("Setting up jlmlcasestudies volume on Modal...")
    print("This will download beyondRGB.zip.005 from Zenodo to /beyondRGB/ subdirectory.")

    result = download_file.remote()
    print(f"Result: {result}")
    print("Volume setup complete!")
