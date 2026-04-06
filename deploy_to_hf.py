"""Deploy Grok Imagine Studio to Hugging Face Spaces.

Usage:
    HF_TOKEN=<token> FAL_KEY=<key> python deploy_to_hf.py
"""

import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
FAL_KEY = os.environ["FAL_KEY"]
SPACE_ID = "FINAL-Bench/Grok-Imagine-Studio"

api = HfApi(token=HF_TOKEN)

# Create space (no-op if exists)
try:
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    print(f"Space {SPACE_ID} ready.")
except Exception as e:
    print(f"Space creation note: {e}")

# Set FAL_KEY secret
api.add_space_secret(repo_id=SPACE_ID, key="FAL_KEY", value=FAL_KEY)
print("FAL_KEY secret set.")

# Upload files
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=SPACE_ID,
    repo_type="space",
)
print("app.py uploaded.")

api.upload_file(
    path_or_fileobj="requirements_hf.txt",
    path_in_repo="requirements.txt",
    repo_id=SPACE_ID,
    repo_type="space",
)
print("requirements.txt uploaded.")

print(f"\nDone! Visit: https://huggingface.co/spaces/{SPACE_ID}")
