"""Download and validate the Equifax enterprise case-file PDFs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"
ALLOWED_SOURCE_DOMAINS = {
    "oversight.house.gov",
    "www.gao.gov",
    "www.ftc.gov",
    "www.fca.org.uk",
}
ALLOWED_RETRIEVAL_DOMAINS = ALLOWED_SOURCE_DOMAINS | {"web.archive.org"}
MIN_EXTRACTED_CHARACTERS = 5_000


def validated_https_domain(url: str, allowed_domains: set[str]) -> str:
    parsed = urlparse(url)
    domain = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or domain not in allowed_domains:
        raise ValueError(f"Unapproved dataset URL: {url}")
    return domain


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "source-highlighter-rag dataset downloader/1.0",
            "Accept": "application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def validate_pdf(path: Path, expected_pages: int, expected_sha256: str) -> None:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"{path.name} does not have a PDF signature.")

    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path.name} checksum mismatch: expected {expected_sha256}, "
            f"received {actual_sha256}."
        )

    reader = PdfReader(path)
    if len(reader.pages) != expected_pages:
        raise ValueError(
            f"{path.name} page-count mismatch: expected {expected_pages}, "
            f"received {len(reader.pages)}."
        )

    extracted_characters = sum(
        len(page.extract_text() or "") for page in reader.pages
    )
    if extracted_characters < MIN_EXTRACTED_CHARACTERS:
        raise ValueError(
            f"{path.name} contains only {extracted_characters} extractable characters."
        )


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    filenames = [document["filename"] for document in documents]
    if len(filenames) != len(set(filenames)):
        raise ValueError("The dataset manifest contains duplicate filenames.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="equifax-dataset-", dir=ROOT) as temp_dir:
        staging_dir = Path(temp_dir)

        for document in documents:
            validated_https_domain(
                document["source_url"],
                ALLOWED_SOURCE_DOMAINS,
            )
            validated_https_domain(
                document["retrieval_url"],
                ALLOWED_RETRIEVAL_DOMAINS,
            )
            staged_path = staging_dir / document["filename"]
            print(f"Downloading {document['filename']}")
            download(document["retrieval_url"], staged_path)
            validate_pdf(
                staged_path,
                document["page_count"],
                document["sha256"],
            )

        for document in documents:
            staged_path = staging_dir / document["filename"]
            destination = DATA_DIR / document["filename"]
            install_path = DATA_DIR / f".{document['filename']}.installing"
            shutil.copyfile(staged_path, install_path)
            install_path.replace(destination)

    print(
        f"Validated {len(documents)} PDFs and "
        f"{sum(document['page_count'] for document in documents)} pages."
    )


if __name__ == "__main__":
    main()
