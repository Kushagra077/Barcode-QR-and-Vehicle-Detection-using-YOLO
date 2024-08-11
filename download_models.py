import os
from huggingface_hub import hf_hub_download, login

# Authenticate using your Hugging Face token
access_token = "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"
login(token=access_token)

# Define the Hugging Face repository and file names
repo_id = "Kushagra77/yolo"  # Your private repository

# List of model filenames to download
models = [
    "yolov8_barcode.pt",
    "yolov8_vehicle.pt",
    "yolov9_barcode.pt",
    "yolov9_vehicle.pt",
    "yolov10_barcode.pt",
    "yolov10_vehicle.pt"
]

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download each model
for model in models:
    print(f"Downloading {model}...")
    file_path = hf_hub_download(repo_id=repo_id, filename=model, cache_dir="models", use_auth_token=True)
    os.rename(file_path, os.path.join("models", model))
    print(f"{model} downloaded successfully!")

print("All models have been downloaded and saved to the 'models' directory.")

