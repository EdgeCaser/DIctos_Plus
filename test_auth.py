from dotenv import load_dotenv, find_dotenv
import os
from huggingface_hub import hf_hub_download
import torch

def test_auth():
    load_dotenv(find_dotenv(), override=True)
    token = os.getenv('HF_TOKEN')
    print(f"Token starts with: {token[:6]}...")
    
    try:
        print("Testing authentication...")
        # Try to download the config file directly
        config_path = hf_hub_download(
            repo_id="pyannote/speaker-diarization",
            filename="config.yaml",
            token=token
        )
        print(f"Success! Config downloaded to: {config_path}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_auth()
