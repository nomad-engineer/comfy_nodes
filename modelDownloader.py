import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse
import folder_paths


class ModelDownloader:
    """
    A ComfyUI node that downloads model files to the ComfyUI models directory.
    Supports CSV format: folder, url, custom_filename
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_list": ("STRING", {
                    "multiline": True,
                    "default": "# Format: folder, url, optional_filename\n# Examples:\n# Single file: checkpoints, https://example.com/model.safetensors\n# Single file with custom name: loras, https://example.com/lora.safetensors, my_lora.safetensors\n# Folder: loras, https://server/models/loras/"
                }),
                "force_redownload": ("BOOLEAN", {"default": False}),
                "recursive_download": ("BOOLEAN", {"default": False}),
                "disable_download": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download_models"
    CATEGORY = "z_mynodes"
    OUTPUT_NODE = True

    def parse_url_entry(self, line):
        """Parse a CSV line into folder, url, and optional filename."""
        parts = [p.strip() for p in line.strip().split(',')]
        if len(parts) < 2:
            return None

        folder = parts[0]
        url = parts[1]
        custom_filename = parts[2] if len(parts) > 2 else None

        return folder, url, custom_filename

    def get_filename_from_url(self, url):
        """Extract filename from URL, removing query parameters."""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if '?' in filename:
            filename = filename.split('?')[0]
        return filename

    def is_folder_url(self, url):
        """
        Determine if a URL points to a folder/directory rather than a file.
        Heuristics:
        - Ends with /
        - No file extension in the last path component
        """
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')

        # If URL explicitly ends with /, it's a folder
        if parsed.path.endswith('/'):
            return True

        # Check if the last component has a file extension
        filename = os.path.basename(path)
        if '.' in filename:
            # Has extension, likely a file
            return False
        else:
            # No extension, likely a folder
            return True

    def get_model_dir(self, folder_type):
        """Get the ComfyUI models directory for a specific folder type."""
        # Try to get the folder path from ComfyUI's folder_paths
        try:
            paths = folder_paths.get_folder_paths(folder_type)
            if paths:
                return Path(paths[0])
        except:
            pass

        # Fallback: use models directory relative to ComfyUI
        models_dir = Path(folder_paths.models_dir) / folder_type
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def download_file_with_temp(self, url, dest_path):
        """
        Download a file to a temporary location and move it to the destination on success.
        This prevents partial/corrupted files from appearing as valid downloads.
        """
        # Create a temporary file in the same directory as the destination
        # This ensures we're on the same filesystem for atomic rename
        dest_dir = dest_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        temp_file = None
        try:
            # Create temp file with a unique name in the destination directory
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=dest_dir,
                prefix='.download_',
                suffix='.tmp'
            ) as tf:
                temp_file = Path(tf.name)

            print(f"  → Downloading to temporary file...")

            # Try aria2c first (faster, multi-connection)
            success = self.download_with_aria2c(url, temp_file)

            # Fall back to wget if aria2c is not available
            if success is None:
                print(f"  → aria2c not found, falling back to wget...")
                success = self.download_with_wget(url, temp_file)

            if success:
                # Verify the downloaded file is not empty
                if temp_file.stat().st_size == 0:
                    print(f"  ✗ Downloaded file is empty")
                    return False

                # Move the temp file to the final destination
                shutil.move(str(temp_file), str(dest_path))
                print(f"  ✓ Successfully downloaded to {dest_path}")
                return True
            else:
                print(f"  ✗ Download failed")
                return False

        except Exception as e:
            print(f"  ✗ Error during download: {e}")
            return False
        finally:
            # Clean up temp file if it still exists
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

    def download_with_aria2c(self, url, dest_path):
        """Download using aria2c (fast, multi-connection)."""
        try:
            result = subprocess.run([
                'aria2c',
                '-x', '16',  # 16 connections
                '-s', '16',  # 16 segments
                '-c',  # continue
                '--file-allocation=none',
                '--max-tries=0',
                '--retry-wait=5',
                '--timeout=60',
                '--allow-overwrite=true',
                '--auto-file-renaming=false',
                '-o', dest_path.name,
                '-d', str(dest_path.parent),
                url
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  aria2c error: {e}")
            return False
        except FileNotFoundError:
            return None  # Signal to try wget

    def download_with_wget(self, url, dest_path):
        """Download using wget."""
        try:
            result = subprocess.run([
                'wget',
                '-c',  # continue
                '-t', '3',  # 3 retries
                '--timeout=30',
                '--read-timeout=30',
                '--progress=bar:force',
                url,
                '-O', str(dest_path)
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  wget error: {e}")
            return False
        except FileNotFoundError:
            print(f"  Error: wget not found. Please install wget or aria2c.")
            return False

    def download_folder(self, url, dest_dir, recursive=False, force_redownload=False):
        """
        Download an entire folder from a URL using wget recursive download.
        Downloads to a temp directory first, then moves files to final location.
        """
        # Model file extensions to download
        model_extensions = [
            'safetensors', 'ckpt', 'pt', 'pth', 'bin',
            'onnx', 'pb', 'h5', 'tflite', 'msgpack'
        ]

        # Create temp directory for download
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix='model_download_')
            temp_path = Path(temp_dir)

            print(f"  → Downloading folder contents to temp directory...")

            # Build wget command
            wget_cmd = [
                'wget',
                '-r',  # recursive
                '--no-parent',  # don't ascend to parent directory
                '--no-host-directories',  # don't create host directory
                '--cut-dirs=999',  # flatten directory structure
                '-t', '3',  # 3 retries
                '--timeout=30',
                '--read-timeout=30',
                '--progress=bar:force',
                '-P', str(temp_path),  # download to temp directory
            ]

            # Add recursive flag
            if not recursive:
                wget_cmd.append('-l')  # level
                wget_cmd.append('1')  # only 1 level (no subdirectories)

            # Add accept pattern for model files
            accept_pattern = ','.join([f'*.{ext}' for ext in model_extensions])
            wget_cmd.append('-A')
            wget_cmd.append(accept_pattern)

            # Reject common non-model files
            wget_cmd.append('-R')
            wget_cmd.append('index.html*,*.tmp,*.txt,*.md')

            wget_cmd.append(url)

            # Execute wget
            result = subprocess.run(wget_cmd, check=True, capture_output=True, text=True)

            # Find all downloaded model files
            downloaded_files = []
            for ext in model_extensions:
                downloaded_files.extend(temp_path.rglob(f'*.{ext}'))

            if not downloaded_files:
                print(f"  ⚠ No model files found in folder")
                return 0, 0

            # Move files from temp to destination
            dest_dir.mkdir(parents=True, exist_ok=True)
            success_count = 0
            skip_count = 0

            for temp_file in downloaded_files:
                filename = temp_file.name
                dest_file = dest_dir / filename

                # Check if file already exists
                if dest_file.exists() and not force_redownload:
                    print(f"  → {filename}: already exists, skipping")
                    skip_count += 1
                    continue

                # Move file to destination
                try:
                    if dest_file.exists():
                        dest_file.unlink()  # Remove existing file if force_redownload
                    shutil.move(str(temp_file), str(dest_file))
                    print(f"  ✓ {filename}: downloaded successfully")
                    success_count += 1
                except Exception as e:
                    print(f"  ✗ {filename}: failed to move - {e}")

            return success_count, skip_count

        except subprocess.CalledProcessError as e:
            print(f"  ✗ wget recursive download failed: {e}")
            return 0, 0
        except FileNotFoundError:
            print(f"  ✗ Error: wget not found. Please install wget.")
            return 0, 0
        except Exception as e:
            print(f"  ✗ Error during folder download: {e}")
            return 0, 0
        finally:
            # Clean up temp directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    def download_models(self, model_list, force_redownload, recursive_download, disable_download):
        """Main function to process and download models."""

        if disable_download:
            return ("Downloads disabled by user",)

        # Parse the model list
        lines = [line.strip() for line in model_list.split('\n')
                 if line.strip() and not line.strip().startswith('#')]

        if not lines:
            return ("No models to download",)

        results = []
        success_count = 0
        skip_count = 0
        fail_count = 0

        print(f"\n{'='*60}")
        print(f"Model Downloader - Processing {len(lines)} entries")
        print(f"Force redownload: {force_redownload}")
        print(f"Recursive download: {recursive_download}")
        print(f"{'='*60}\n")

        for line in lines:
            result = self.parse_url_entry(line)
            if not result:
                print(f"⚠ Skipping invalid entry: {line}")
                fail_count += 1
                continue

            folder, url, custom_filename = result

            # Get destination directory
            dest_dir = self.get_model_dir(folder)

            # Check if URL is a folder or a file
            if self.is_folder_url(url):
                # Folder download
                print(f"Processing folder: {url}")
                print(f"  Target folder: {folder}")
                print(f"  Destination: {dest_dir}")

                folder_success, folder_skip = self.download_folder(
                    url, dest_dir, recursive=recursive_download, force_redownload=force_redownload
                )

                success_count += folder_success
                skip_count += folder_skip

                if folder_success == 0 and folder_skip == 0:
                    fail_count += 1

            else:
                # Single file download
                # Determine filename
                if custom_filename:
                    filename = custom_filename
                else:
                    filename = self.get_filename_from_url(url)

                dest_file = dest_dir / filename

                print(f"Processing file: {filename}")
                print(f"  Target folder: {folder}")
                print(f"  Destination: {dest_file}")

                # Check if file exists
                if dest_file.exists() and not force_redownload:
                    print(f"  ✓ File already exists, skipping")
                    skip_count += 1
                    print()
                    continue

                if dest_file.exists() and force_redownload:
                    print(f"  ⚠ File exists but force_redownload is enabled")
                    # Remove existing file
                    try:
                        dest_file.unlink()
                        print(f"  → Removed existing file")
                    except Exception as e:
                        print(f"  ✗ Failed to remove existing file: {e}")
                        fail_count += 1
                        print()
                        continue

                # Download the file
                success = self.download_file_with_temp(url, dest_file)

                if success:
                    success_count += 1
                else:
                    fail_count += 1

            print()  # Empty line between entries

        # Summary
        print(f"{'='*60}")
        print(f"Download Summary:")
        print(f"  ✓ Successfully downloaded: {success_count}")
        print(f"  → Skipped (already exists): {skip_count}")
        print(f"  ✗ Failed: {fail_count}")
        print(f"{'='*60}\n")

        status = f"Downloaded: {success_count}, Skipped: {skip_count}, Failed: {fail_count}"
        return (status,)
