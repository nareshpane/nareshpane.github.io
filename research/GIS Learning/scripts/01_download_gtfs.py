"""Download and extract an immutable, dated TransLink GTFS snapshot."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests


SOURCE_URL = "https://gtfs-static.translink.ca/gtfs/google_transit.zip"
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
EXTRACTED_DIR = PROJECT_DIR / "data" / "extracted"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"


def checksum(path: Path) -> tuple[str, int]:
    """Return the SHA-256 and byte size of an existing snapshot."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def download_snapshot(target: Path) -> tuple[str, int]:
    """Stream the official archive to disk and return its SHA-256 and size."""
    temporary = target.with_suffix(target.suffix + ".part")
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing snapshot: {target}. "
            "Remove it explicitly only if you intend to replace the evidence file."
        )
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0

    try:
        with requests.get(
            SOURCE_URL,
            stream=True,
            timeout=(15, 180),
            headers={"User-Agent": "Metro-Vancouver-GTFS-learning-project/1.0"},
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        if size == 0:
            raise RuntimeError("The GTFS download completed but contained zero bytes.")
        with ZipFile(temporary) as gtfs_zip:
            if damaged_member := gtfs_zip.testzip():
                raise RuntimeError(f"Downloaded ZIP contains a damaged member: {damaged_member}")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return digest.hexdigest(), size


def extract_snapshot(archive: Path, destination: Path) -> list[str]:
    """Extract the archive without changing it, rejecting unsafe member paths."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite extracted snapshot: {destination}")
    if destination.exists():
        destination.rmdir()
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete extraction directory exists: {temporary}. "
            "Inspect and remove it explicitly before retrying."
        )
    temporary.mkdir(parents=True, exist_ok=False)

    try:
        with ZipFile(archive) as gtfs_zip:
            members = [item for item in gtfs_zip.infolist() if not item.is_dir()]
            root = temporary.resolve()
            for member in members:
                target = (temporary / member.filename).resolve()
                if root not in target.parents:
                    raise RuntimeError(f"Unsafe path in GTFS archive: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with gtfs_zip.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        temporary.replace(destination)
    except BadZipFile as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {archive}") from exc
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return sorted(member.filename for member in members)


def write_metadata(
    *, snapshot_date: str, timestamp: str, filename: str, size: int, sha256: str
) -> None:
    """Write snapshot provenance and a conventional checksum file."""
    metadata = {
        "dataset_name": "TransLink GTFS Static Data",
        "source_organization": "TransLink",
        "source_url": SOURCE_URL,
        "download_date": snapshot_date,
        "download_timestamp": timestamp,
        "filename": filename,
        "file_size_bytes": size,
        "sha256": sha256,
        "geographic_coverage": "Metro Vancouver, British Columbia, Canada",
        "original_format": "GTFS Static ZIP archive",
        "source_note": "Official TransLink GTFS Static Data",
    }
    (METADATA_DIR / "source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (METADATA_DIR / "sha256.txt").write_text(
        f"{sha256}  data/raw/{filename}\n", encoding="utf-8"
    )


def main() -> None:
    """Download, verify, extract, and describe today's GTFS snapshot."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    source_file = METADATA_DIR / "source.json"
    if source_file.exists():
        recorded = json.loads(source_file.read_text(encoding="utf-8"))
        archive = RAW_DIR / str(recorded["filename"])
        snapshot_date = str(recorded["download_date"])
        destination = EXTRACTED_DIR / snapshot_date
        if not archive.exists():
            raise FileNotFoundError(f"Metadata records a missing raw snapshot: {archive}")
        sha256, size = checksum(archive)
        if sha256 != recorded["sha256"] or size != recorded["file_size_bytes"]:
            raise RuntimeError("The retained raw snapshot does not match source.json.")
        print(f"Using verified retained snapshot: {archive.name}")
        if destination.exists() and any(destination.iterdir()):
            files = [path for path in destination.rglob("*") if path.is_file()]
            print(f"Using existing extraction: {len(files)} files in {destination}")
        else:
            files = extract_snapshot(archive, destination)
            print(f"Extracted {len(files)} files to {destination}")
        print(f"SHA-256: {sha256}")
        return

    downloaded_at = datetime.now().astimezone()
    snapshot_date = downloaded_at.date().isoformat()
    filename = f"translink_gtfs_{snapshot_date}.zip"
    archive = RAW_DIR / filename
    destination = EXTRACTED_DIR / snapshot_date

    print(f"Downloading {SOURCE_URL}")
    sha256, size = download_snapshot(archive)
    files = extract_snapshot(archive, destination)
    write_metadata(
        snapshot_date=snapshot_date,
        timestamp=downloaded_at.isoformat(timespec="seconds"),
        filename=filename,
        size=size,
        sha256=sha256,
    )
    print(f"Saved immutable snapshot: {archive.name} ({size:,} bytes)")
    print(f"SHA-256: {sha256}")
    print(f"Extracted {len(files)} files to {destination}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, requests.RequestException, RuntimeError) as exc:
        raise SystemExit(f"GTFS download failed: {exc}") from exc
