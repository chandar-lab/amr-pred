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

# Zenodo record of the released models (the number at the end of its DOI, 10.5281/zenodo.<record>)
ZENODO_RECORD_ID = None
ARCHIVE_NAME = "amr_pred_models.zip"
TOP_DIR = "amr_pred_models"  # folder at the root of the archive; its contents go to --out


def get_parser():
    parser = argparse.ArgumentParser(description="Download the released models from Zenodo.")
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "models"),
                        help="Directory to extract the models to (default: models/ in the repository).")
    parser.add_argument("--record", type=str, default=ZENODO_RECORD_ID,
                        help="Zenodo record ID to download from (default: the record of the paper's models).")
    parser.add_argument("--archive", type=str, default=None,
                        help=f"Extract a local copy of {ARCHIVE_NAME} instead of downloading it.")
    parser.add_argument("--force", action="store_true", help="Replace the models already in --out.")
    return parser


def zenodo_file(record):
    """URL and checksum ("md5:<hex>") of ARCHIVE_NAME in the Zenodo record."""
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
        files = json.load(response)["files"]
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
            if member.is_dir():
                continue
            parts = member.filename.split("/")
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
            if args.record is None:
                sys.exit("No Zenodo record is set yet; pass --record <id>, or --archive with a local copy of the models.")
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
