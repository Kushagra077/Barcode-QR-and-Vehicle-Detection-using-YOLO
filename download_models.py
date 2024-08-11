import os
import requests

# Your Hugging Face token
access_token = "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"

# List of URLs and corresponding filenames
models = {
    "yolov10_barcode.pt": "https://huggingface.co/Kushagra77/yolo/resolve/main/yolov10_barcode.pt",
    "yolov10_vehicle.pt": "https://huggingface.co/Kushagra77/yolo/resolve/main/yolov10_vehicle.pt",
    "yolov8_barcode.pt": "https://huggingface.co/Kushagra77/yolo/resolve/main/yolov8_barcode.pt",
    "yolov8_vehicle.pt": "https://huggingface.co/Kushagra77/yolo/resolve/main/yolov8_vehicle.pt",
    "yolov9_barcode.pt": "https://huggingface.co/Kushagra77/yolo/resolve/main/yolov9_barcode.pt",
    "yolov9_vehicle.pt": "https://huggingface.co/Kushagra77/yolo/resolve/main/yolov9_vehicle.pt"
}

# Directory to save the models
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def download_model(url, filename):
    """Download a model file from a URL and save it to the specified filename."""
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(models_dir, filename)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded successfully: {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

# Download all models
for filename, url in models.items():
    download_model(url, filename)

print("All models have been downloaded.")
