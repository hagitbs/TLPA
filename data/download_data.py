from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from google.api_core.exceptions import NotFound  # type: ignore
from google.cloud import secretmanager, storage  # type: ignore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"


def get_secrets(secrets: List[str]) -> Dict[str, str]:
    secret_client = secretmanager.SecretManagerServiceClient()
    res = {}
    for secret_name in secrets:
        name = secret_client.secret_version_path(
            "temporal-dynamics", secret_name, "latest"
        )
        response = secret_client.access_secret_version(name)
        res[secret_name] = response.payload.data.decode("utf-8")
    return res


def download(bucket, blob_name: str, destination: Path):
    blob = bucket.blob(blob_name)
    destination_path = destination / blob_name
    try:
        blob.reload(timeout=15)
    except NotFound:
        print("file doesn't exist in cloud storage")
        raise FileNotFoundError
    if not destination_path.exists():
        blob.download_to_filename(destination_path)
        print(f"finished downloading {blob_name}. thank you for your patience :)")
    else:
        if os.path.getsize(destination_path) != blob.size:
            blob.download_to_filename(destination_path)
            print(
                f"finished downloading a newer version of {blob_name}. thank you for your patience :)"
            )
        else:
            print(
                f"skipped downloading {destination_path.name}, since it already exists"
            )
            return


def download_folder(bucket, folder):
    blobs = bucket.list_blobs(prefix=folder)
    for blob in blobs:
        download(bucket, blob.name, Path("loco"))


if __name__ == "__main__":
    client = storage.Client()
    bucket = client.bucket("loco_data")
    download(bucket, "metadata.csv", Path("loco"))
    download_folder(bucket, "np_freq/")
