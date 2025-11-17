import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import folder_paths
import requests
from tqdm import tqdm
import comfy.utils
import comfy.model_management
import fnmatch
import re

# Try to import BeautifulSoup, but don't fail if it's not available
try:
    from bs4 import BeautifulSoup

    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    print(
        "Warning: BeautifulSoup4 not found. Wildcard matching will use regex fallback."
    )
    print("For better wildcard support, install: pip install beautifulsoup4")


class ModelDownloader:
    """
    A ComfyUI node that downloads model files to the ComfyUI models directory.
    Supports CSV format: folder, url, custom_filename

    Features:
    - Single file downloads
    - Folder downloads (recursive or non-recursive)
    - Wildcard URL patterns (e.g., https://server/path/*.safetensors)
    - Progress bar in ComfyUI UI
    - Cancellation support
    - HTTP authentication (username/password)
    - Concurrent downloads
    """

    def __init__(self):
        self.cancel_requested = False
        self.progress_bar = None
        self.username = ""
        self.password = ""

    def check_interruption(self):
        """Check if ComfyUI workflow has been interrupted."""
        if comfy.model_management.processing_interrupted():
            self.cancel_requested = True
        return self.cancel_requested

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_list": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "# Format: folder, url, optional_filename\n# Examples:\n# Single file: checkpoints, https://example.com/model.safetensors\n# Single file with custom name: loras, https://example.com/lora.safetensors, my_lora.safetensors\n# Folder: loras, https://server/models/loras/\n# Wildcard: loras, https://server/models/loras/*.safetensors",
                    },
                ),
                "concurrent_downloads": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "force_redownload": ("BOOLEAN", {"default": False}),
                "recursive_download": ("BOOLEAN", {"default": False}),
                "disable_download": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "username": ("STRING", {"default": ""}),
                "password": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download_models"
    CATEGORY = "z_mynodes"
    OUTPUT_NODE = True

    def parse_url_entry(self, line):
        """Parse a CSV line into folder, url, and optional filename."""
        parts = [p.strip() for p in line.strip().split(",")]
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
        if "?" in filename:
            filename = filename.split("?")[0]
        return filename

    def has_wildcard(self, url):
        """Check if a URL contains wildcard characters (* or ?)."""
        return "*" in url or "?" in url

    def find_wildcard_matches(self, url_with_wildcard):
        """
        Performs a wildcard search on a remote HTTP directory listing to find matching files.
        Returns list of tuples: (filename, full_url, size)
        """
        print(f"\n--- Wildcard URL Detection ---")
        print(f"Processing wildcard URL: {url_with_wildcard}")

        # Separate the base URL from the filename pattern
        path_index = url_with_wildcard.rfind("/")
        if path_index == -1:
            print("Error: Invalid URL format. Could not find path separator.")
            return []

        base_url = url_with_wildcard[: path_index + 1]
        pattern = url_with_wildcard[path_index + 1 :]

        if not pattern or ("*" not in pattern and "?" not in pattern):
            print(f"Warning: No valid wildcard found in pattern: '{pattern}'.")
            return []

        print(f"  Base Directory URL: {base_url}")
        print(f"  Wildcard Pattern: {pattern}")

        # Fetch the directory listing HTML
        try:
            headers = {"User-Agent": "ComfyUI-ModelDownloader/1.0"}
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            response = requests.get(base_url, headers=headers, auth=auth, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {base_url}: {e}")
            return []

        # Parse the HTML content
        matching_files = []

        # Use BeautifulSoup if available, otherwise fall back to regex
        if HAS_BEAUTIFULSOUP:
            soup = BeautifulSoup(response.content, "html.parser")
            links = []
            for link in soup.find_all("a"):
                href = link.get("href")
                if href:
                    # Handle both string and list cases
                    if isinstance(href, list):
                        links.extend(href)
                    else:
                        links.append(str(href))
        else:
            # Regex fallback for HTML parsing
            href_pattern = r'href=["\']([^"\']+)["\']'
            links = re.findall(href_pattern, response.text)

        for href in links:
            # Skip invalid links
            if (
                not href
                or href in [".", "..", "/", "#"]
                or href.startswith("?")
                or href.startswith("#")
            ):
                continue

            # URL decode and extract filename
            from urllib.parse import unquote

            decoded_href = unquote(str(href))
            filename = decoded_href.split("/")[-1].split("?")[0]

            # Skip if no filename or doesn't match pattern
            if not filename or not fnmatch.fnmatch(filename, pattern):
                continue

            # Construct full URL - handle different href formats
            href_str = str(href)
            if href_str.startswith("http://") or href_str.startswith("https://"):
                full_url = href_str
            elif href_str.startswith("/"):
                # Absolute path - construct from base URL domain
                parsed_base = urlparse(base_url)
                full_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href_str}"
            else:
                # Relative path - join with base URL properly
                full_url = urljoin(base_url, href_str)

            print(f"  ✓ Match: {filename}")
            print(f"    href='{href_str}' -> {full_url}")

            size = self.get_file_size(full_url)
            size_str = f"{size / (1024 * 1024):.2f} MB" if size > 0 else "unknown size"
            print(f"    Size: {size_str}")

            matching_files.append((filename, full_url, size))

        print(f"  Total matches: {len(matching_files)}")
        return matching_files

    def is_folder_url(self, url):
        """
        Determine if a URL points to a folder/directory rather than a file.
        Heuristics:
        - Ends with /
        - No file extension in the last path component
        - Does not contain wildcards (wildcards are handled separately)
        """
        # Wildcard URLs are not folder URLs
        if self.has_wildcard(url):
            return False

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        # If URL explicitly ends with /, it's a folder
        if parsed.path.endswith("/"):
            return True

        # Check if the last component has a file extension
        filename = os.path.basename(path)
        if "." in filename:
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

    def get_file_size(self, url):
        """Get the size of a file from URL headers."""
        try:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            response = requests.head(url, allow_redirects=True, timeout=10, auth=auth)
            response.raise_for_status()
            if "content-length" in response.headers:
                return int(response.headers["content-length"])
            # If HEAD doesn't return content-length, try GET with stream
            response = requests.get(
                url, allow_redirects=True, timeout=10, auth=auth, stream=True
            )
            response.raise_for_status()
            if "content-length" in response.headers:
                return int(response.headers["content-length"])
        except Exception as e:
            # Silently fail - size is optional
            pass
        return 0

    def get_folder_files_listing(self, url, recursive=False):
        """
        Get a list of model files from a folder URL.
        Returns list of tuples: (filename, url, estimated_size)
        """
        model_extensions = [
            "safetensors",
            "ckpt",
            "pt",
            "pth",
            "bin",
            "onnx",
            "pb",
            "h5",
            "tflite",
            "msgpack",
        ]

        files = []
        try:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            response = requests.get(url, timeout=30, auth=auth)
            response.raise_for_status()

            # Find all href links in the HTML
            href_pattern = r'href=["\']([^"\']+)["\']'
            hrefs = re.findall(href_pattern, response.text, re.IGNORECASE)

            for href in hrefs:
                # Skip invalid links
                if not href or href.startswith("?") or href.startswith("#"):
                    continue

                # Check if link ends with a model extension
                href_lower = href.lower()
                if any(href_lower.endswith(f".{ext}") for ext in model_extensions):
                    # Construct full URL
                    if href.startswith("http"):
                        file_url = href
                    elif href.startswith("/"):
                        parsed = urlparse(url)
                        file_url = f"{parsed.scheme}://{parsed.netloc}{href}"
                    else:
                        file_url = url.rstrip("/") + "/" + href.lstrip("/")

                    filename = os.path.basename(file_url.split("?")[0])
                    size = self.get_file_size(file_url)
                    files.append((filename, file_url, size))

        except Exception:
            pass

        return files

    def collect_all_download_tasks(self, lines, force_redownload):
        """
        Collect all files that need to be downloaded.
        Returns list of dicts with: folder, filename, url, dest_path, size
        """
        tasks = []

        for line in lines:
            result = self.parse_url_entry(line)
            if not result:
                continue

            folder, url, custom_filename = result
            dest_dir = self.get_model_dir(folder)

            if self.has_wildcard(url):
                # Wildcard URL - find all matching files
                print(f"\nProcessing wildcard pattern for folder '{folder}'")
                wildcard_files = self.find_wildcard_matches(url)
                for filename, file_url, size in wildcard_files:
                    dest_path = dest_dir / filename
                    if not dest_path.exists() or force_redownload:
                        tasks.append(
                            {
                                "folder": folder,
                                "filename": filename,
                                "url": file_url,
                                "dest_path": dest_path,
                                "size": size,
                                "is_wildcard_match": True,
                            }
                        )
            elif self.is_folder_url(url):
                # Folder - get all files from it
                folder_files = self.get_folder_files_listing(url)
                for filename, file_url, size in folder_files:
                    dest_path = dest_dir / filename
                    if not dest_path.exists() or force_redownload:
                        tasks.append(
                            {
                                "folder": folder,
                                "filename": filename,
                                "url": file_url,
                                "dest_path": dest_path,
                                "size": size,
                                "is_folder_item": True,
                            }
                        )
            else:
                # Single file
                filename = (
                    custom_filename
                    if custom_filename
                    else self.get_filename_from_url(url)
                )
                dest_path = dest_dir / filename
                if not dest_path.exists() or force_redownload:
                    size = self.get_file_size(url)
                    tasks.append(
                        {
                            "folder": folder,
                            "filename": filename,
                            "url": url,
                            "dest_path": dest_path,
                            "size": size,
                            "is_folder_item": False,
                        }
                    )

        return tasks

    def download_file_with_temp(self, url, dest_path, progress_callback=None):
        """
        Download a file to a temporary location and move it to the destination on success.
        This prevents partial/corrupted files from appearing as valid downloads.
        """
        # Check for cancellation
        if self.check_interruption():
            print(f"  ⚠ Download cancelled")
            return False

        # Create a temporary file in the same directory as the destination
        # This ensures we're on the same filesystem for atomic rename
        dest_dir = dest_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        temp_file = None
        process = None
        try:
            # Create temp file with a unique name in the destination directory
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=dest_dir,
                prefix=".download_",
                suffix=".tmp",
            ) as tf:
                temp_file = Path(tf.name)

            print(f"  → Downloading to temporary file...")
            print(f"  → URL: {url}")

            # Try aria2c first (faster, multi-connection)
            success, process = self.download_with_aria2c(
                url, temp_file, progress_callback
            )

            # Fall back to wget if aria2c is not available
            if success is None:
                print(f"  → aria2c not found, falling back to wget...")
                success, process = self.download_with_wget(
                    url, temp_file, progress_callback
                )

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
            if self.cancel_requested:
                print(f"  ⚠ Download cancelled")
            else:
                print(f"  ✗ Error during download: {e}")

            # Kill the download process if it's still running
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            return False
        finally:
            # Clean up temp file if it still exists
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

    def download_with_aria2c(self, url, dest_path, progress_callback=None):
        """Download using aria2c (fast, multi-connection)."""
        try:
            cmd = [
                "aria2c",
                "-x",
                "16",
                "-s",
                "16",  # 16 connections/segments
                "-c",  # continue
                "--file-allocation=none",
                "--max-tries=0",
                "--retry-wait=5",
                "--timeout=60",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
            ]

            # Add authentication if provided
            if self.username and self.password:
                cmd.extend(
                    ["--http-user", self.username, "--http-passwd", self.password]
                )

            cmd.extend(["-o", dest_path.name, "-d", str(dest_path.parent), url])

            return self._run_download_process(cmd, "aria2c")

        except FileNotFoundError:
            return None, None  # Signal to try wget

    def download_with_wget(self, url, dest_path, progress_callback=None):
        """Download using wget."""
        try:
            cmd = [
                "wget",
                "-c",  # continue
                "-t",
                "3",  # 3 retries
                "--timeout=30",
                "--read-timeout=30",
                "--progress=bar:force",
            ]

            # Add authentication if provided
            if self.username and self.password:
                cmd.extend(["--user", self.username, "--password", self.password])

            cmd.extend([url, "-O", str(dest_path)])

            return self._run_download_process(cmd, "wget")

        except FileNotFoundError:
            print(f"  Error: wget not found. Please install wget or aria2c.")
            return False, None

    def _run_download_process(self, cmd, tool_name):
        """Common method to run download processes with cancellation support."""
        import time

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for completion while checking for cancellation
        while process.poll() is None:
            if self.check_interruption():
                self._terminate_process(process)
                return False, process
            time.sleep(0.1)

        if process.returncode == 0:
            return True, process
        else:
            # Print stderr output for debugging
            stderr_output = (
                process.stderr.read().decode("utf-8", errors="ignore")
                if process.stderr
                else ""
            )
            print(f"  {tool_name} error: return code {process.returncode}")
            if stderr_output:
                # Print first few lines of error
                error_lines = stderr_output.strip().split("\n")[:3]
                for line in error_lines:
                    print(f"    {line}")
            return False, process

    def _terminate_process(self, process):
        """Safely terminate a process."""
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except:
                pass

    def download_with_wget(self, url, dest_path, progress_callback=None):
        """Download using wget."""
        try:
            cmd = [
                "wget",
                "-c",  # continue
                "-t",
                "3",  # 3 retries
                "--timeout=30",
                "--read-timeout=30",
                "--progress=bar:force",
            ]

            # Add authentication if provided
            if self.username and self.password:
                cmd.extend(["--user", self.username])
                cmd.extend(["--password", self.password])

            cmd.extend([url, "-O", str(dest_path)])

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for completion while checking for cancellation
            while process.poll() is None:
                if self.check_interruption():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False, process
                # Small sleep to avoid busy waiting
                import time

                time.sleep(0.1)

            if process.returncode == 0:
                return True, process
            else:
                # Print stderr output for debugging
                stderr_output = (
                    process.stderr.read().decode("utf-8", errors="ignore")
                    if process.stderr
                    else ""
                )
                print(f"  wget error: return code {process.returncode}")
                if stderr_output:
                    # Print first few lines of error
                    error_lines = stderr_output.strip().split("\n")[:5]
                    for line in error_lines:
                        print(f"    {line}")
                return False, process
        except FileNotFoundError:
            print(f"  Error: wget not found. Please install wget or aria2c.")
            return False, None
        except Exception as e:
            print(f"  wget exception: {e}")
            return False, None

    def download_folder(self, url, dest_dir, recursive=False, force_redownload=False):
        """
        Download an entire folder from a URL using wget recursive download.
        Downloads to a temp directory first, then moves files to final location.
        """
        # Model file extensions to download
        model_extensions = [
            "safetensors",
            "ckpt",
            "pt",
            "pth",
            "bin",
            "onnx",
            "pb",
            "h5",
            "tflite",
            "msgpack",
        ]

        # Create temp directory for download
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="model_download_")
            temp_path = Path(temp_dir)

            print(f"  → Downloading folder contents to temp directory...")

            # Build wget command
            wget_cmd = [
                "wget",
                "-r",  # recursive
                "--no-parent",  # don't ascend to parent directory
                "--no-host-directories",  # don't create host directory
                "--cut-dirs=999",  # flatten directory structure
                "-t",
                "3",  # 3 retries
                "--timeout=30",
                "--read-timeout=30",
                "--progress=bar:force",
                "-P",
                str(temp_path),  # download to temp directory
            ]

            # Add recursive flag
            if not recursive:
                wget_cmd.append("-l")  # level
                wget_cmd.append("1")  # only 1 level (no subdirectories)

            # Add accept pattern for model files
            accept_pattern = ",".join([f"*.{ext}" for ext in model_extensions])
            wget_cmd.append("-A")
            wget_cmd.append(accept_pattern)

            # Reject common non-model files
            wget_cmd.append("-R")
            wget_cmd.append("index.html*,*.tmp,*.txt,*.md")

            wget_cmd.append(url)

            # Execute wget
            result = subprocess.run(
                wget_cmd, check=True, capture_output=True, text=True
            )

            # Find all downloaded model files
            downloaded_files = []
            for ext in model_extensions:
                downloaded_files.extend(temp_path.rglob(f"*.{ext}"))

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

    def download_models(
        self,
        model_list,
        concurrent_downloads,
        force_redownload,
        recursive_download,
        disable_download,
        username="",
        password="",
    ):
        """Main function to process and download models."""

        # Reset cancellation flag and set credentials
        self.cancel_requested = False
        self.username = username
        self.password = password

        if disable_download:
            return ("Downloads disabled by user",)

        # Parse the model list
        lines = [
            line.strip()
            for line in model_list.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        if not lines:
            return ("No models to download",)

        print(f"\n{'=' * 60}")
        print(f"Model Downloader - Collecting file listings...")
        print(f"{'=' * 60}\n")

        # Phase 1: Collect all download tasks
        try:
            tasks = self.collect_all_download_tasks(lines, force_redownload)
        except Exception as e:
            print(f"✗ Error collecting download tasks: {e}")
            return ("Failed to collect download tasks",)

        if not tasks:
            print("No files to download (all files already exist)")
            return ("No files to download",)

        # Print full file listing
        print(f"\n{'=' * 60}")
        print(f"Files to download: {len(tasks)}")
        print(f"{'=' * 60}")
        total_size = sum(task["size"] for task in tasks)
        for i, task in enumerate(tasks, 1):
            size_mb = task["size"] / (1024 * 1024) if task["size"] > 0 else 0
            size_str = f"{size_mb:.2f} MB" if size_mb > 0 else "unknown size"
            print(f"{i}. {task['filename']} ({size_str})")
            print(f"   → {task['folder']}")
        print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")
        print(f"Concurrent downloads: {concurrent_downloads}")
        print(f"{'=' * 60}\n")

        # Phase 2: Initialize progress bar
        self.progress_bar = comfy.utils.ProgressBar(len(tasks))

        # Thread-safe printing and progress tracking
        print_lock = threading.Lock()
        progress_lock = threading.Lock()
        completed_count = [0]  # Use list for mutable counter

        # Total counters
        total_success = 0
        total_fail = 0

        def download_single_task(task, task_index):
            """Download a single file and update progress."""
            nonlocal total_success, total_fail

            if self.cancel_requested:
                return False

            with print_lock:
                print(
                    f"\n[{task_index + 1}/{len(tasks)}] Downloading: {task['filename']}"
                )
                print(f"  Target folder: {task['folder']}")
                print(f"  Destination: {task['dest_path']}")

            # Download the file
            success = self.download_file_with_temp(task["url"], task["dest_path"])

            # Update progress
            with progress_lock:
                completed_count[0] += 1
                self.progress_bar.update_absolute(completed_count[0], len(tasks))

            return success

        # Phase 3: Download files
        try:
            if concurrent_downloads == 1:
                # Sequential processing
                for i, task in enumerate(tasks):
                    if self.cancel_requested:
                        print("\n⚠ Download cancelled by user")
                        break

                    success = download_single_task(task, i)
                    if success:
                        total_success += 1
                    else:
                        total_fail += 1
            else:
                # Concurrent processing
                with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
                    # Submit all download tasks
                    futures = {
                        executor.submit(download_single_task, task, i): (task, i)
                        for i, task in enumerate(tasks)
                    }

                    # Collect results as they complete
                    for future in as_completed(futures):
                        if self.cancel_requested:
                            # Cancel remaining tasks
                            for f in futures:
                                f.cancel()
                            print("\n⚠ Download cancelled by user")
                            break

                        try:
                            success = future.result()
                            if success:
                                total_success += 1
                            else:
                                total_fail += 1
                        except Exception as e:
                            with print_lock:
                                print(f"✗ Unexpected error: {e}")
                            total_fail += 1

        except KeyboardInterrupt:
            print("\n⚠ Download interrupted")
            self.cancel_requested = True
        except Exception as e:
            print(f"\n✗ Unexpected error during downloads: {e}")

        # Summary
        print(f"\n{'=' * 60}")
        print(f"Download Summary:")
        print(f"  ✓ Successfully downloaded: {total_success}")
        print(f"  ✗ Failed: {total_fail}")
        if self.cancel_requested:
            print(f"  ⚠ Cancelled: {len(tasks) - total_success - total_fail}")
        print(f"{'=' * 60}\n")

        status = f"Downloaded: {total_success}, Failed: {total_fail}"
        if self.cancel_requested:
            status += f", Cancelled: {len(tasks) - total_success - total_fail}"

        return (status,)
