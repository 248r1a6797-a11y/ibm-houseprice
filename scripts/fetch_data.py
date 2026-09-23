"""Fetch and archive the authoritative Ames Housing dataset and documentation."""
import urllib.request
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CSV_TARGET = RAW_DIR / "AmesHousing.csv"
DOC_TARGET = RAW_DIR / "DataDocumentation.txt"

# Primary & Fallback URLs
CSV_URLS = [
    "https://raw.githubusercontent.com/ds4stats/r-tutorials/master/data-viz/data/AmesHousing.csv",
    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
]

DOC_URLS = [
    "http://jse.amstat.org/v19n3/decock/DataDocumentation.txt",
    "https://raw.githubusercontent.com/ds4stats/r-tutorials/master/data-viz/data/DataDocumentation.txt"
]

def fetch_file(urls, target_path: Path, is_tsv: bool = False):
    """Attempt download from list of URLs."""
    if target_path.exists():
        logger.info(f"Target already exists: {target_path}")
        return True

    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            logger.info(f"Attempting download from {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read()
                if is_tsv and target_path.suffix == ".csv":
                    # Convert TSV to CSV if downloaded from AmesHousing.txt
                    from io import StringIO
                    text = content.decode("utf-8", errors="ignore")
                    df = pd.read_csv(StringIO(text), sep="\t")
                    df.to_csv(target_path, index=False)
                else:
                    with open(target_path, "wb") as f:
                        f.write(content)
            logger.info(f"Successfully saved to {target_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to fetch from {url}: {e}")
    return False

def main():
    logger.info("Starting data ingestion process...")
    csv_ok = fetch_file(CSV_URLS, CSV_TARGET)
    doc_ok = fetch_file(DOC_URLS, DOC_TARGET)

    if not csv_ok:
        raise RuntimeError("Failed to download Ames Housing dataset from all sources.")

    # Validate shape
    df = pd.read_csv(CSV_TARGET)
    logger.info(f"Dataset verification: Shape = {df.shape}, Target 'SalePrice' present = {'SalePrice' in df.columns}")
    assert df.shape[0] == 2930, f"Expected 2930 rows, got {df.shape[0]}"
    assert df.shape[1] >= 81, f"Expected >= 81 columns, got {df.shape[1]}"
    logger.info("Data fetch and integrity verification completed successfully.")

if __name__ == "__main__":
    main()
