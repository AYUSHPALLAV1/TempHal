import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

def upload_to_hf(username: str, repo_name: str, folder_path: str):
    """
    Uploads the fine-tuned classifier to HuggingFace Hub.
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        print("HF_TOKEN not found in .env. Please add it to upload the model.")
        return
        
    api = HfApi(token=token)
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating private repository {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        
    print(f"Uploading folder {folder_path} to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Successfully uploaded model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading folder: {e}")

if __name__ == "__main__":
    # Replace with your actual HF username
    HF_USERNAME = "ayushpallav1"
    REPO_NAME = "temphal-classifier"
    FOLDER_PATH = "outputs/classifier"
    
    upload_to_hf(HF_USERNAME, REPO_NAME, FOLDER_PATH)
