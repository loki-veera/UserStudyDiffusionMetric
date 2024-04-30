import os
import toml

from typing import List
from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image
from io import BytesIO


class GCP_Connection():
    def __init__(self, bucket_name: str, credential_path: str) -> None:
        """Initialize Google Cloud Platform.

        Args:
            bucket_name (str): Bucket name
            credential_path (str): Path to google credentials
        """
        credential_data = None
        if os.path.exists(credential_path):
            with open(credential_path, "r") as file:
                credential_data = toml.load(file)
        if credential_data is None:
            raise RuntimeError("Error connecting to GCP.")
        credentials = service_account.Credentials.from_service_account_info(credential_data["google_cloud"])
        self.storage_client = storage.Client(credentials=credentials)
        self.bucket = self.storage_client.get_bucket(bucket_name)
    

    def write_csv(self, file_name: str) -> None:
        """Write CSV file to GCP platform.

        Args:
            file_name (str): File name
        """
        blob = self.bucket.blob(f"user_responses_pjfgan/{file_name}")
        blob.upload_from_filename(file_name)


    def get_image_names(self, prefix: str) -> List:
        """Get image names for specific prefix.

        Args:
            prefix (str): Specific prefix

        Returns:
            List: List containing the image names
        """
        img_names = [blob.name for blob in self.bucket.list_blobs(prefix=prefix)]
        return img_names[1:] # First one is prefix itself.
    

    def open_image(self, img_name: str) -> Image.Image:
        """Load image from GCP.

        Args:
            img_name (str): Image name

        Returns:
            Image.Image: Opened image
        """
        blob = self.bucket.blob(img_name)
        blob_content = blob.download_as_bytes()
        img = Image.open(BytesIO(blob_content))
        return img


    def get_num_files(self, prefix: str) -> int:
        """Given directory return number of files.

        Args:
            prefix (str): Directory

        Returns:
            int: Number of files
        """
        # Remove one becuase its directory itself
        return sum(1 for _ in self.bucket.list_blobs(prefix=prefix))-1
