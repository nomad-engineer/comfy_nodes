#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

def parse_url_entry(line):
    """Parse a CSV line into folder, url, and optional filename."""
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) < 2:
        return None

    folder = parts[0]
    url = parts[1]
    custom_filename = parts[2] if len(parts) > 2 else None

    return folder, url, custom_filename

def get_filename_from_url(url):
    """Extract filename from URL, removing query parameters."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if '?' in filename:
        filename = filename.split('?')[0]
    return filename

def get_remote_file_size(url):
    """Get the size of the remote file without downloading it."""
    try:
        # Try with a HEAD request first
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except:
        pass

    # Fallback to GET request with range to get headers
    try:
        req = urllib.request.Request(url)
        req.add_header('Range', 'bytes=0-0')
        with urllib.request.urlopen(req, timeout=10) as response:
            content_range = response.headers.get('Content-Range')
            if content_range:
                # Format is "bytes 0-0/total_size"
                total_size = content_range.split('/')[-1]
                return int(total_size)

            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except:
        pass

    return None

def verify_file(dest_path, url):
    """Verify if local file exists and matches remote file size."""
    if not dest_path.exists():
        return False, "File does not exist"

    local_size = dest_path.stat().st_size

    if local_size == 0:
        return False, "File is empty"

    print(f"    Verifying file size...", end='', flush=True)
    remote_size = get_remote_file_size(url)

    if remote_size is None:
        print(f" (cannot determine remote size, assuming OK)")
        # If we can't get remote size, assume file is OK if it's not empty
        return True, "Cannot verify size"

    if local_size == remote_size:
        print(f" ✓ Size matches ({local_size:,} bytes)")
        return True, "Size matches"
    else:
        print(f" ✗ Size mismatch (local: {local_size:,}, remote: {remote_size:,})")
        return False, f"Size mismatch: local {local_size:,} bytes, remote {remote_size:,} bytes"

def download_file(url, dest_path):
    """Download a file using wget with resume capability."""
    try:
        # Use wget with:
        # -c: continue/resume partial downloads
        # -t 0: infinite retries
        # --timeout=30: 30 second timeout
        # --waitretry=5: wait 5 seconds between retries
        # --read-timeout=30: 30 second read timeout
        # -O: output file
        result = subprocess.run([
            'wget',
            '-c',  # Continue partial downloads
            '-t', '0',  # Infinite retries
            '--timeout=30',
            '--waitretry=5',
            '--read-timeout=30',
            '--progress=bar:force',
            url,
            '-O', str(dest_path)
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  Error downloading: {e}")
        return False
    except FileNotFoundError:
        print("\n  Error: wget not found. Please install wget.")
        return False

def download_file_aria2c(url, dest_dir, filename):
    """Download a file using aria2c (faster, better for large files)."""
    try:
        # Use aria2c with:
        # -x 16: use 16 connections
        # -s 16: split into 16 segments
        # -c: continue partial downloads
        # --file-allocation=none: don't preallocate (faster start)
        # -d: output directory
        # -o: output filename
        result = subprocess.run([
            'aria2c',
            '-x', '16',
            '-s', '16',
            '-c',
            '--file-allocation=none',
            '--max-tries=0',
            '--retry-wait=5',
            '--timeout=60',
            '-d', str(dest_dir),
            '-o', filename,
            url
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  Error downloading: {e}")
        return False
    except FileNotFoundError:
        print("\n  Error: aria2c not found. Falling back to wget.")
        return None  # Signal to try wget

def process_downloads(entries, output_dir, use_aria2c=False, force_redownload=False):
    """Process all download entries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        result = parse_url_entry(entry)
        if not result:
            print(f"Skipping invalid entry: {entry}")
            continue

        folder, url, custom_filename = result

        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = get_filename_from_url(url)

        # Create destination directory
        dest_dir = output_path / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_file = dest_dir / filename

        # Check if file exists and verify size
        should_download = True
        if dest_file.exists() and not force_redownload:
            print(f"  Checking {filename}...")
            is_valid, message = verify_file(dest_file, url)

            if is_valid:
                print(f"  ✓ {filename} already exists and is complete, skipping")
                should_download = False
            else:
                print(f"  ⚠ {filename} exists but is incomplete: {message}")
                print(f"  → Re-downloading {filename}...")

        if should_download:
            if not dest_file.exists():
                print(f"  → Downloading {filename} to {dest_dir}/")

            success = False
            if use_aria2c:
                result = download_file_aria2c(url, dest_dir, filename)
                if result is None:  # aria2c not found
                    success = download_file(url, dest_file)
                else:
                    success = result
            else:
                success = download_file(url, dest_file)

            if success:
                # Verify the download
                is_valid, message = verify_file(dest_file, url)
                if is_valid:
                    print(f"  ✓ Downloaded and verified {filename}")
                else:
                    print(f"  ⚠ Downloaded {filename} but verification failed: {message}")
            else:
                print(f"  ✗ Failed to download {filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Download models from URLs to organized directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From file
  %(prog)s downloads.csv

  # From stdin
  cat downloads.csv | %(prog)s -

  # Use aria2c for faster downloads
  %(prog)s downloads.csv --aria2c

  # Force re-download all files
  %(prog)s downloads.csv --force

  # Custom output directory
  %(prog)s downloads.csv -o /custom/models
        """
    )
    parser.add_argument('input', help='CSV file path, "-" for stdin, or comma-separated string')
    parser.add_argument('-o', '--output-dir', default='/models',
                       help='Output directory (default: /models)')
    parser.add_argument('--aria2c', action='store_true',
                       help='Use aria2c for faster multi-connection downloads')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')

    args = parser.parse_args()

    entries = []

    # Determine input source
    if args.input == '-':
        # Read from stdin
        entries = [line.strip() for line in sys.stdin if line.strip()]
    elif ',' in args.input and not os.path.exists(args.input):
        # Treat as direct CSV string
        entries = [args.input]
    else:
        # Read from file
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

        with open(args.input, 'r') as f:
            entries = [line.strip() for line in f if line.strip()]

    if not entries:
        print("No entries to download")
        sys.exit(1)

    print(f"Processing {len(entries)} download(s) to {args.output_dir}")
    process_downloads(entries, args.output_dir, use_aria2c=args.aria2c, force_redownload=args.force)
    print("Done!")

if __name__ == '__main__':
    main()
