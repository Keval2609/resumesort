
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("download_models")

MODELS_TO_DOWNLOAD = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

def main():
    log.info("Starting model pre-download...")
    for model_name in MODELS_TO_DOWNLOAD:
        log.info(f"Downloading/Caching model: {model_name}")
        try:
            # Loading the model forces it to download and cache locally
            model = SentenceTransformer(model_name)
            log.info(f"Successfully cached: {model_name}")
        except Exception as e:
            log.error(f"Failed to download {model_name}: {e}")
            raise e

    log.info("All models downloaded and cached successfully.")

if __name__ == "__main__":
    main()
