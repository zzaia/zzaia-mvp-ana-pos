"""Azure Blob Storage data fetch utilities for the ANA-POS pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from azure_utils import AzureBlobDownloader


def _is_valid_pdf(path: Path) -> bool:
    """
    Check whether a file starts with the PDF magic bytes.

    Args:
        path: Path to the file to check.

    Returns:
        True if the file begins with ``%PDF``, False otherwise.
    """
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def fetch_dataset(azure_client: AzureBlobDownloader | None, pdf_path: Path) -> None:
    """
    Ensure the Sumulas PDF is present and valid, downloading from Azure when needed.

    Args:
        azure_client: Initialized ``AzureBlobDownloader`` instance, or None for offline mode.
        pdf_path: Local destination path for the PDF file.

    Raises:
        FileNotFoundError: If the PDF is absent and ``azure_client`` is None.
        ValueError: If the downloaded blob is not a valid PDF.
        Exception: Re-raised from the Azure SDK on download failure.
    """
    if pdf_path.exists() and _is_valid_pdf(pdf_path):
        print(f"Dataset already present: {pdf_path}")
        return
    if pdf_path.exists():
        pdf_path.unlink()
        print("Removed corrupt local PDF — re-downloading...")
    if azure_client is None:
        raise FileNotFoundError(
            f"Dataset not found at {pdf_path} and Azure client is unavailable."
            " Upload 'Sumulas - STJ.pdf' to ./datasets/ manually."
        )
    print("Downloading dataset from Azure...")
    blob_path = "datasets/S\u00famulas - STJ.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    blob_client = azure_client.container_client.get_blob_client(blob_path)
    data = blob_client.download_blob().readall()
    if data[:4] != b"%PDF":
        raise ValueError(
            f"Blob at {blob_path!r} is not a valid PDF ({data[:64]!r})."
            " Check the container path and re-upload the dataset."
        )
    with open(pdf_path, "wb") as f:
        f.write(data)
    print(f"Dataset downloaded: {pdf_path}")


def fetch_checkpoints(azure_client: AzureBlobDownloader | None, checkpoint_dir: Path) -> None:
    """
    Ensure pipeline checkpoints are present locally, downloading from Azure when needed.

    Args:
        azure_client: Initialized ``AzureBlobDownloader`` instance, or None for offline mode.
        checkpoint_dir: Local directory where ``.pkl`` checkpoint files are stored.
    """
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pkl")):
        print(f"Checkpoints already present: {checkpoint_dir}")
        return
    if azure_client is None:
        print("Azure client unavailable — checkpoints not found, pipeline will run from scratch.")
        return
    print("Downloading checkpoints from Azure...")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for blob in azure_client.container_client.list_blobs(name_starts_with="checkpoints/"):
        if not blob.name.endswith(".pkl"):
            continue
        local_path = checkpoint_dir / Path(blob.name).name
        bc = azure_client.container_client.get_blob_client(blob.name)
        with open(local_path, "wb") as f:
            f.write(bc.download_blob().readall())
        downloaded += 1
    print(f"Checkpoints downloaded: {downloaded} files")
