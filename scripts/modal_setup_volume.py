"""
Modal script to create the jlmlcasestudies volume and download beyondRGB data from Zenodo.

This script:
1. Creates a Modal volume named 'jlmlcasestudies' (or uses existing one)
2. Downloads all 5 parts of the beyondRGB dataset from Zenodo to a 'beyondRGB' subdirectory
3. Can inspect the contents of the combined split zip archive
"""

import modal

# Constants
# Zenodo URLs for all 5 parts of the split zip archive
ZENODO_BASE_URL = "https://zenodo.org/records/16848482/files/beyondRGB.zip"
ZIP_PARTS = [".001", ".002", ".003", ".004", ".005"]
DOWNLOAD_TIMEOUT = 86400  # 24 hours timeout for large download

# Create Modal app
app = modal.App("setup-jlmlcasestudies-volume")

# Create or get the volume
volume = modal.Volume.from_name("jlmlcasestudies", create_if_missing=True)

# Define the image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install("curl", "wget", "p7zip-full")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=86400,  # 24 hour timeout for large download
)
def download_all_parts():
    """
    Download all 5 parts of the beyondRGB dataset from Zenodo to the volume.
    """
    import subprocess
    import os
    import shutil

    # Clean up any old files at root level from previous runs
    for part in ZIP_PARTS:
        old_file = f"/vol/beyondRGB.zip{part}"
        if os.path.exists(old_file):
            print(f"Removing old file at root: {old_file}")
            os.remove(old_file)

    # Create the target directory
    target_dir = "/vol/beyondRGB"

    # Clean up existing directory if it exists to avoid mixing old and new files
    if os.path.exists(target_dir):
        print(f"Cleaning up existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    downloaded_files = []
    total_size = 0

    # Download each part
    for part in ZIP_PARTS:
        url = f"{ZENODO_BASE_URL}{part}?download=1"
        target_file = f"{target_dir}/beyondRGB.zip{part}"

        print(f"\n{'='*60}")
        print(f"Downloading part {part}...")
        print(f"URL: {url}")
        print(f"Target: {target_file}")
        print("=" * 60)

        # Use wget with progress
        result = subprocess.run(
            [
                "wget",
                "--progress=dot:mega",  # Show progress every MB
                "-c",  # Continue/resume if interrupted
                "-O", target_file,
                url
            ],
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
        )

        if result.returncode != 0:
            print(f"Download failed for {part}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(
                f"Download failed for {part} with return code {result.returncode}. "
                f"stderr: {result.stderr}"
            )

        # Check file size
        if os.path.exists(target_file):
            file_size = os.path.getsize(target_file)
            total_size += file_size
            downloaded_files.append(target_file)
            print(f"Downloaded {part}: {file_size} bytes ({file_size / (1024**3):.2f} GB)")
        else:
            raise RuntimeError(f"Download failed - file not found at {target_file}")

    # Commit the volume changes
    volume.commit()
    print("\nVolume committed successfully")

    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total parts downloaded: {len(downloaded_files)}")
    print(f"Total size: {total_size} bytes ({total_size / (1024**3):.2f} GB)")
    for f in downloaded_files:
        print(f"  - {f}")

    return f"Success: Downloaded {len(downloaded_files)} parts ({total_size / (1024**3):.2f} GB total)"


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,  # 10 minute timeout for inspection
)
def inspect_zip():
    """
    Inspect the contents of the split zip archive without extracting.
    Uses 7z to list contents of the split archive.
    """
    import subprocess
    import os

    target_dir = "/vol/beyondRGB"
    first_part = f"{target_dir}/beyondRGB.zip.001"

    if not os.path.exists(first_part):
        return f"Error: First part not found at {first_part}"

    print(f"Inspecting split archive starting from: {first_part}")
    print("=" * 60)

    # List all parts present
    print("\nParts found:")
    for part in ZIP_PARTS:
        part_path = f"{target_dir}/beyondRGB.zip{part}"
        if os.path.exists(part_path):
            size = os.path.getsize(part_path)
            print(f"  {part}: {size} bytes ({size / (1024**3):.2f} GB)")
        else:
            print(f"  {part}: MISSING")

    # Use 7z to list contents of the split archive
    print("\n" + "=" * 60)
    print("Archive contents (using 7z l):")
    print("=" * 60)

    result = subprocess.run(
        ["7z", "l", first_part],
        capture_output=True,
        text=True,
        timeout=300,
    )

    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")

    return result.stdout


@app.local_entrypoint()
def main():
    """
    Main entrypoint for running the volume setup on Modal.
    Downloads all 5 parts from Zenodo, then inspects the archive contents.
    """
    print("Setting up jlmlcasestudies volume on Modal...")
    print(
        "This will download all 5 parts of the beyondRGB dataset from Zenodo."
    )
    print(
        "WARNING: This download will take several hours depending on bandwidth."
    )
    print("=" * 60)

    # Download all parts
    result = download_all_parts.remote()
    print(f"Download result: {result}")
    
    # Inspect the archive
    print("\n" + "=" * 60)
    print("Inspecting archive contents...")
    print("=" * 60)
    inspect_result = inspect_zip.remote()
    print(f"Inspection complete.")
    
    print("\nVolume setup complete!")
