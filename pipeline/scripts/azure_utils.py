"""
Utility functions for dataset processing and Azure blob storage operations.
"""

from pathlib import Path
from azure.storage.blob import BlobServiceClient


class AzureBlobDownloader:
    """
    Azure Blob Storage downloader class for managing dataset downloads.

    This class encapsulates Azure blob operations and maintains connection state
    for efficient dataset management operations.
    """

    def __init__(self, account_url, container_name):
        """
        Initialize Azure blob downloader.

        Args:
            account_url (str): Azure storage account URL
            container_name (str): Name of the blob container

        Raises:
            Exception: If connection to Azure fails or Azure SDK not available
        """
        try:
            self.account_url = account_url
            self.container_name = container_name
            self.blob_service_client = BlobServiceClient(account_url=account_url)
            self.container_client = self.blob_service_client.get_container_client(container_name)

        except Exception as e:
            raise Exception(f"Failed to initialize Azure Blob connection: {e}")


    def download_documents(self, project_folder, document_folder, base_path="../"):
        """
        Download dataset from Azure Blob Storage.

        Args:
            project_folder: Name of the project folder in blob storage
            document_folder: Name of the document folder in blob storage
            base_path: Local base path for downloads (default: "../")

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            original_dir = Path(base_path) / project_folder
            specific_dir = original_dir / document_folder

            original_dir.mkdir(exist_ok=True)
            specific_dir.mkdir(exist_ok=True)

            remote_path = f"{project_folder}/{document_folder}/"
            downloaded_files = 0

            for blob in self.container_client.list_blobs(name_starts_with=remote_path):
                blob_client = self.container_client.get_blob_client(blob.name)
                local_file_path = Path(base_path) / blob.name
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                blob_data = blob_client.download_blob()
                with open(local_file_path, "wb") as download_file:
                    download_file.write(blob_data.readall())
                downloaded_files += 1

            print(f"Successfully downloaded {downloaded_files} files from Azure Blob Storage")
            return True

        except Exception as e:
            print(f"Failed to download from Azure Blob Storage: {e}")
            return False
