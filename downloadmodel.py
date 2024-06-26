from huggingface_hub import hf_hub_download
import time
import requests

HUGGING_FACE_API_KEY = "YOUR_API_KEY"  # Make sure to keep API keys secure and not expose them unnecessarily.

model_id = "dslim/bert-base-NER"
filenames = [
    "onnx/added_tokens.json","onnx/config.json","onnx/model.onnx",
    "onnx/special_tokens_map.json","onnx/tokenizer.json",
    "onnx/tokenizer_config.json","onnx/vocab.txt","added_tokens.json", 
    "config.json", "flax_model.msgpack", "model.safetensors", 
    "pytorch_model.bin", "special_tokens_map.json","tf_model.h5",
    "tokenizer_config.json","onnx/vocab.txt"
]

def download_file(filename, max_retries=5):
    for attempt in range(max_retries):
        try:
            downloaded_model_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                token=HUGGING_FACE_API_KEY
            )
            print(downloaded_model_path)
            break  # Break the loop if the download was successful
        except requests.exceptions.RequestException as e:  # Catch network-related errors
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    else:
        print(f"Failed to download {filename} after {max_retries} attempts.")

for filename in filenames:
    download_file(filename)


