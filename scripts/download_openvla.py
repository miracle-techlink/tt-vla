#!/usr/bin/env python3
"""Download OpenVLA shards with multiple strategies."""
import os, sys, time, subprocess

HF_MIRROR = "https://hf-mirror.com"
REPO = "openvla/openvla-7b"
SHARDS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--openvla--openvla-7b")
BLOB_DIR = os.path.join(CACHE, "blobs")
SNAP_DIR = os.path.join(CACHE, "snapshots")
MANUAL_DIR = os.path.join(CACHE, "manual_shards")
os.makedirs(MANUAL_DIR, exist_ok=True)

# Clean incomplete
for f in os.listdir(BLOB_DIR):
    if f.endswith(".incomplete"):
        os.remove(os.path.join(BLOB_DIR, f))
        print(f"Removed incomplete: {f}")

def download_with_wget(shard, dest):
    url = f"{HF_MIRROR}/{REPO}/resolve/main/{shard}"
    print(f"\n[wget] Downloading {shard}...", flush=True)
    cmd = [
        "wget", "-c", "--no-check-certificate",
        "--timeout=300", "--tries=10", "--waitretry=5",
        "-O", dest, url
    ]
    proc = subprocess.run(cmd, capture_output=False)
    return proc.returncode == 0

def download_with_python(shard, dest):
    import requests
    url = f"{HF_MIRROR}/{REPO}/resolve/main/{shard}"
    print(f"\n[python] Downloading {shard}...", flush=True)

    headers = {}
    downloaded = 0
    if os.path.exists(dest):
        downloaded = os.path.getsize(dest)
        headers["Range"] = f"bytes={downloaded}-"
        print(f"  Resuming from {downloaded/1e9:.2f}GB", flush=True)

    try:
        r = requests.get(url, stream=True, timeout=(30, 600),
                        allow_redirects=True, headers=headers)
        if r.status_code == 416:
            print(f"  Already complete!", flush=True)
            return True
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0)) + downloaded
        start = time.time()
        mode = "ab" if downloaded > 0 else "wb"

        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start
                speed = (downloaded - (total - int(r.headers.get("content-length", 0)))) / max(elapsed, 1) / 1024 / 1024
                pct = downloaded / total * 100 if total else 0
                print(f"\r  {downloaded/1e9:.2f}GB / {total/1e9:.2f}GB ({pct:.1f}%) - {speed:.1f} MB/s    ", end="", flush=True)

        print(f"\n  Done!", flush=True)
        return True
    except Exception as e:
        print(f"\n  Error: {e}", flush=True)
        return False

def download_with_hf_transfer(shard, dest):
    """Use huggingface_hub with hf_transfer."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"\n[hf_transfer] Downloading {shard}...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            REPO, shard,
            local_dir=MANUAL_DIR,
            force_download=True,
        )
        print(f"  Done: {path}", flush=True)
        return True
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        return False

# Main download loop
for shard in SHARDS:
    dest = os.path.join(MANUAL_DIR, shard)

    # Check if already downloaded
    if os.path.exists(dest) and os.path.getsize(dest) > 1e9:
        print(f"SKIP {shard} ({os.path.getsize(dest)/1e9:.1f}GB)", flush=True)
        continue

    # Try strategies in order
    success = False

    # Strategy 1: wget with resume
    success = download_with_wget(shard, dest)

    if not success or not os.path.exists(dest) or os.path.getsize(dest) < 1e9:
        # Strategy 2: python requests with resume
        success = download_with_python(shard, dest)

    if not success or not os.path.exists(dest) or os.path.getsize(dest) < 1e9:
        # Strategy 3: hf_transfer
        success = download_with_hf_transfer(shard, dest)

# Now link shards into HF cache structure
print("\n\nLinking shards into HF cache...", flush=True)
snap_dirs = os.listdir(SNAP_DIR) if os.path.exists(SNAP_DIR) else []
if snap_dirs:
    snap = os.path.join(SNAP_DIR, snap_dirs[0])
    for shard in SHARDS:
        src = os.path.join(MANUAL_DIR, shard)
        dst = os.path.join(snap, shard)
        if os.path.exists(src) and os.path.getsize(src) > 1e9:
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            print(f"  Linked: {shard}", flush=True)

# Verify
print("\nVerification:", flush=True)
for shard in SHARDS:
    path = os.path.join(MANUAL_DIR, shard)
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f"  {shard}: {sz/1e9:.2f}GB {'OK' if sz > 1e9 else 'TOO SMALL'}", flush=True)
    else:
        print(f"  {shard}: MISSING", flush=True)

print("\nDone!", flush=True)
