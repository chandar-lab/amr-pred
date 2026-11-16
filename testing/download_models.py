"""
Download the trained models released with the paper from Zenodo, and extract them into models/ in the repository.

The models are XGBoost models (default parameters, best k-mer size, 1M k-mers) for the 9 species x antibiotic
combinations, each with its k-mer groups and metadata, in models/<species>__<antibiotic>/ (see models/README.md).

Usage (from the repository root):
    python testing/download_models.py [--out models] [--archive <local amr_pred_models.zip>] [--force]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Zenodo record of the released models, all versions: https://doi.org/10.5281/zenodo.24285925
# (the latest version is downloaded; --record <id of a version> picks a specific one)
ZENODO_RECORD_ID = "24285925"
ARCHIVE_NAME = "amr_pred_models.zip"
TOP_DIR = "amr_pred_models"  # folder at the root of the archive; its contents go to --out


def get_parser():
    parser = argparse.ArgumentParser(description="Download the released models from Zenodo.")
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "models"),
                        help="Directory to extract the models to (default: models/ in the repository).")
    parser.add_argument("--record", type=str, default=None,
                        help="Zenodo record ID of a specific version to download (default: the latest version of the "
                        "paper's models).")
    parser.add_argument("--archive", type=str, default=None,
                        help=f"Extract a local copy of {ARCHIVE_NAME} instead of downloading it.")
    parser.add_argument("--force", action="store_true", help="Replace the models already in --out.")
    return parser


def zenodo_file(record=None):
    """URL and checksum ("md5:<hex>") of ARCHIVE_NAME in a Zenodo record (default: the latest version of the models)."""
    url = f"https://zenodo.org/api/records/{record}" if record else \
        f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/versions/latest"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    files = data["files"]
    print(f"Zenodo record {data['id']} (doi:{data['doi']})")
    for entry in files:
        if entry["key"] == ARCHIVE_NAME:
            return entry["links"]["self"], entry["checksum"]
    raise FileNotFoundError(f"Zenodo record {record} has no file named {ARCHIVE_NAME}: {[f['key'] for f in files]}")


def download(url, checksum, path):
    """Download url to path, checking its checksum."""
    algorithm, expected = checksum.split(":", 1)
    digest = hashlib.new(algorithm)
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as f:
        size = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(2**20):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if size:
                print(f"\r  {done / 2**20:.1f} / {size / 2**20:.1f} MB", end="", flush=True)
    print()
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {url}: expected {expected}, got {digest.hexdigest()}")


def extract(archive, out):
    """Extract the files under TOP_DIR/ in the archive into out, refusing paths that leave out."""
    out = os.path.abspath(out)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = member.filename.split("/")
            # Skip directories, and the metadata macOS adds when zipping (__MACOSX/, .DS_Store)
            if member.is_dir() or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store":
                continue
            if parts[0] != TOP_DIR or len(parts) < 2:
                raise ValueError(f"Unexpected file in {archive}: {member.filename}")
            target = os.path.abspath(os.path.join(out, *parts[1:]))
            if not target.startswith(out + os.sep):
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def check_sha256sums(out):
    """Check the extracted files against the SHA256SUMS file of the release."""
    with open(os.path.join(out, "SHA256SUMS")) as f:
        for line in f:
            expected, name = line.rstrip("\n").split("  ", 1)
            with open(os.path.join(out, name), "rb") as g:
                if hashlib.sha256(g.read()).hexdigest() != expected:
                    raise ValueError(f"SHA-256 mismatch for {name}")


def main():
    args = get_parser().parse_args()
    if os.path.isdir(args.out) and os.listdir(args.out) and not args.force:
        sys.exit(f"{args.out} is not empty; use --force to replace its models.")

    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive
        if archive is None:
            archive = os.path.join(tmp, ARCHIVE_NAME)
            download(*zenodo_file(args.record), archive)
        if args.force and os.path.isdir(args.out):
            shutil.rmtree(args.out)
        extract(archive, args.out)
    check_sha256sums(args.out)

    models = sorted(d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)))
    print(f"Extracted {len(models)} models to {args.out}: {', '.join(models)}")


if __name__ == "__main__":
    main()
