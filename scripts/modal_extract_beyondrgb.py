"""
Extract files from beyondRGB.zip.001 on Modal volume.

This script extracts files from the partial ZIP archive on the Modal volume,
mirroring the directory structure from the ZIP file into the beyondRGB directory.

The beyondRGB.zip.001 is a multi-part ZIP archive. Since we only have the first part,
we can't use standard unzip tools (they need the central directory from the last part).
Instead, we parse local file headers and decompress files sequentially.
"""

import modal

# Create Modal app
app = modal.App("extract-beyondrgb")

# Reference the existing volume
volume = modal.Volume.from_name("jlmlcasestudies")

# Define image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py")


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=30 * 60,  # 30 minute timeout
)
def extract_files():
    """
    Extract files from beyondRGB.zip.001 to the beyondRGB directory on the volume.
    
    Parses local file headers from the ZIP and extracts complete files,
    mirroring the directory structure.
    """
    import os
    import struct
    import zlib
    import h5py
    
    zip_path = "/data/beyondRGB/beyondRGB.zip.001"
    output_base = "/data/beyondRGB"
    
    print(f"Reading ZIP file: {zip_path}")
    
    # Check file size
    file_size = os.path.getsize(zip_path)
    print(f"ZIP file size: {file_size / (1024**3):.2f} GB")
    
    # Read the file
    with open(zip_path, 'rb') as f:
        data = f.read()
    
    print(f"Read {len(data):,} bytes")
    
    # Parse local file headers
    pos = 0
    files_extracted = 0
    files_truncated = 0
    directories_created = set()
    
    while pos < len(data) - 30:
        # Check for local file header signature: PK\x03\x04
        sig = struct.unpack('<I', data[pos:pos+4])[0]
        if sig != 0x04034b50:
            # Not a local file header, try next byte (shouldn't happen in well-formed ZIP)
            pos += 1
            continue
        
        # Parse local file header
        version = struct.unpack('<H', data[pos+4:pos+6])[0]
        flags = struct.unpack('<H', data[pos+6:pos+8])[0]
        compression = struct.unpack('<H', data[pos+8:pos+10])[0]
        mod_time = struct.unpack('<H', data[pos+10:pos+12])[0]
        mod_date = struct.unpack('<H', data[pos+12:pos+14])[0]
        crc32 = struct.unpack('<I', data[pos+14:pos+18])[0]
        comp_size = struct.unpack('<I', data[pos+18:pos+22])[0]
        uncomp_size = struct.unpack('<I', data[pos+22:pos+26])[0]
        name_len = struct.unpack('<H', data[pos+26:pos+28])[0]
        extra_len = struct.unpack('<H', data[pos+28:pos+30])[0]
        
        # Get filename
        name = data[pos+30:pos+30+name_len].decode('utf-8', errors='replace')
        
        # Calculate file data location
        file_data_start = pos + 30 + name_len + extra_len
        file_data_end = file_data_start + comp_size
        
        # Check if file is complete in our partial download
        is_complete = file_data_end <= len(data)
        
        # Check if encrypted
        is_encrypted = bool(flags & 0x1)
        
        # Determine output path (remove 'beyond-unzip/' prefix if present)
        if name.startswith('beyond-unzip/'):
            relative_path = name[len('beyond-unzip/'):]
        else:
            relative_path = name
        
        output_path = os.path.join(output_base, relative_path)
        
        # Skip if it's a directory entry (ends with /)
        if name.endswith('/'):
            if output_path not in directories_created:
                os.makedirs(output_path, exist_ok=True)
                directories_created.add(output_path)
            pos = file_data_start
            continue
        
        # Skip incomplete files
        if not is_complete:
            print(f"  TRUNCATED: {name} ({uncomp_size:,} bytes)")
            files_truncated += 1
            break  # Once we hit a truncated file, all subsequent files will also be truncated
        
        # Extract complete files
        if comp_size > 0:
            # Get compressed data
            compressed_data = data[file_data_start:file_data_end]
            
            # Decompress if needed
            if compression == 8:  # DEFLATE
                try:
                    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                    file_content = decompressor.decompress(compressed_data) + decompressor.flush()
                except Exception as e:
                    print(f"  ERROR decompressing {name}: {e}")
                    pos = file_data_end
                    continue
            elif compression == 0:  # Stored (no compression)
                file_content = compressed_data
            else:
                print(f"  UNSUPPORTED compression method {compression} for {name}")
                pos = file_data_end
                continue
            
            # Verify size
            if len(file_content) != uncomp_size:
                print(f"  SIZE MISMATCH for {name}: got {len(file_content)}, expected {uncomp_size}")
                pos = file_data_end
                continue
            
            # Create output directory
            output_dir = os.path.dirname(output_path)
            if output_dir and output_dir not in directories_created:
                os.makedirs(output_dir, exist_ok=True)
                directories_created.add(output_dir)
            
            # Write file
            with open(output_path, 'wb') as out_f:
                out_f.write(file_content)
            
            print(f"  Extracted: {relative_path} ({len(file_content):,} bytes)")
            files_extracted += 1
        
        # Move to next entry
        pos = file_data_end if comp_size > 0 else file_data_start
    
    # Commit the volume changes
    volume.commit()
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Files extracted: {files_extracted}")
    print(f"  Files truncated (incomplete): {files_truncated}")
    print(f"  Directories created: {len(directories_created)}")
    
    # Verify some H5 files can be read
    print(f"\nVerifying extracted HDF5 files...")
    h5_files = []
    for root, dirs, files in os.walk(output_base):
        for f in files:
            if f.endswith('.h5'):
                h5_files.append(os.path.join(root, f))
    
    for h5_path in h5_files[:5]:  # Verify first 5
        try:
            with h5py.File(h5_path, 'r') as hf:
                keys = list(hf.keys())
                print(f"  ✓ {os.path.relpath(h5_path, output_base)}: keys={keys}")
        except Exception as e:
            print(f"  ✗ {os.path.relpath(h5_path, output_base)}: {e}")
    
    return {
        'files_extracted': files_extracted,
        'files_truncated': files_truncated,
        'directories_created': len(directories_created),
        'h5_files_found': len(h5_files)
    }


@app.local_entrypoint()
def main():
    """Run the extraction on Modal."""
    print("Starting extraction of beyondRGB.zip.001 on Modal...")
    result = extract_files.remote()
    print(f"\nResults: {result}")
