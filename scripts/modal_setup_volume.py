"""
Modal script to create the jlmlcasestudies volume and download beyondRGB data from OneDrive.

This script:
1. Creates a Modal volume named 'jlmlcasestudies' (or uses existing one)
2. Downloads the full beyondRGB dataset (~200GB) from OneDrive to a 'beyondRGB' subdirectory
"""

import modal

# Constants
# OneDrive sharing link - converted to direct download format
ONEDRIVE_SHARE_URL = "https://1drv.ms/u/s!AheBo1Cre0p_gYhXovaSSrG3LNo1Pg?e=V17jcM"
DOWNLOAD_TIMEOUT = 86400  # 24 hours timeout for large download

# Create Modal app
app = modal.App("setup-jlmlcasestudies-volume")

# Create or get the volume
volume = modal.Volume.from_name("jlmlcasestudies", create_if_missing=True)

# Define the image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install("curl", "wget")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=86400,  # 24 hour timeout for large download
)
def download_file():
    """
    Download the full beyondRGB dataset from OneDrive to the volume.
    """
    import subprocess
    import os
    import shutil
    import base64

    # Clean up any old files at root level from previous runs
    for old_file in ["/vol/beyondRGB.zip.005"]:
        if os.path.exists(old_file):
            print(f"Removing old file: {old_file}")
            os.remove(old_file)

    # Create the target directory
    target_dir = "/vol/beyondRGB"

    # Clean up existing directory if it exists to avoid mixing old and new files
    if os.path.exists(target_dir):
        print(f"Cleaning up existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    # Convert OneDrive sharing link to direct download URL
    # OneDrive share links can be converted by changing the base URL and encoding
    share_url = ONEDRIVE_SHARE_URL
    
    # Extract the share ID and convert to direct download format
    # The format is: https://api.onedrive.com/v1.0/shares/{encoded_url}/root/content
    encoded_url = base64.urlsafe_b64encode(share_url.encode()).decode().rstrip("=")
    direct_url = f"https://api.onedrive.com/v1.0/shares/u!{encoded_url}/root/content"
    
    print(f"OneDrive share URL: {share_url}")
    print(f"Direct download URL: {direct_url}")

    # Download the file - use wget for better large file handling
    target_file = f"{target_dir}/beyondRGB.zip"

    print(f"\nDownloading full beyondRGB dataset (~200GB)...")
    print("This will take a long time. Progress will be shown below.")
    
    # Use wget with progress and resume capability
    result = subprocess.run(
        [
            "wget",
            "--progress=dot:giga",  # Show progress every GB
            "-c",  # Continue/resume if interrupted
            "-O", target_file,
            direct_url
        ],
        capture_output=False,  # Show progress in real-time
        timeout=DOWNLOAD_TIMEOUT,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Download failed with return code {result.returncode}")

    # Check file size
    if os.path.exists(target_file):
        file_size = os.path.getsize(target_file)
        print(f"\nDownloaded {file_size} bytes ({file_size / (1024**3):.2f} GB) to {target_file}")
    else:
        raise RuntimeError(f"Download failed - file not found at {target_file}")

    # Commit the volume changes
    volume.commit()
    print("\nVolume committed successfully")

    return f"Success: Downloaded {target_file} ({file_size / (1024**3):.2f} GB)"


@app.local_entrypoint()
def main():
    """
    Main entrypoint for running the volume setup on Modal.
    """
    print("Setting up jlmlcasestudies volume on Modal...")
    print(
        "This will download the full beyondRGB dataset (~200GB) from OneDrive."
    )
    print(
        "WARNING: This download will take several hours depending on bandwidth."
    )

    result = download_file.remote()
    print(f"Result: {result}")
    print("Volume setup complete!")
