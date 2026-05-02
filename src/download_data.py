import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_from_kaggle():
    # Step 1: Check kaggle.json
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(
            f"kaggle.json not found at {kaggle_path}\n"
            "Download it from: https://www.kaggle.com/settings → API → Create New Token"
        )

    # Step 2: Authenticate
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    # Step 3: Download dataset
    print("Downloading dataset...")
    api.dataset_download_files(
        "paultimothymooney/chest-xray-pneumonia",
        path=".",
        unzip=False
    )

    zip_path = "chest-xray-pneumonia.zip"

    if not os.path.exists(zip_path):
        raise FileNotFoundError("Download failed. ZIP file not found.")

    # Step 4: Extract
    print("Extracting dataset...")
    os.makedirs("data", exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/")

    # Step 5: Remove nested chest_xray folder if it exists and move files up
    nested_path = "data/chest_xray/chest_xray"
    if os.path.exists(nested_path):
        print("Flattening nested folder structure...")
        import shutil
        # Move contents from nested folder to parent
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join("data/chest_xray", item)
            if os.path.exists(dst) and os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        # Remove empty nested folder
        shutil.rmtree(nested_path)

    # Step 6: Remove macOS metadata
    macosx_path = "data/chest_xray/__MACOSX"
    if os.path.exists(macosx_path):
        print("Removing macOS metadata...")
        import shutil
        shutil.rmtree(macosx_path)

    # Step 7: Cleanup
    os.remove(zip_path)

    print("✅ Done! Data is ready in /data")


if __name__ == "__main__":
    download_from_kaggle()