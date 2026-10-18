"""Fetch the Wind Turbine SCADA dataset (Kaggle, berkerisen) and verify it.

Primary source (needs a free Kaggle login):
    https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset
    -> download, unzip, place T1.csv in data/raw/

This script fetches a public GitHub mirror of the same file and checks its
MD5, so the run is reproducible without Kaggle credentials. Two independent
mirrors were confirmed byte-identical when this project was built.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://raw.githubusercontent.com/dominodatalab/reference-project-wind-turbine-scada/main/data/T1.csv",
    "https://raw.githubusercontent.com/VidyaMS/wind_turbine_data_analysis/main/T1.csv",
]
MD5 = "61655b9229603bd68fd206e00ef18bd8"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "T1.csv"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and md5(DEST) == MD5:
        print(f"OK, already present and verified: {DEST}")
        return
    for url in MIRRORS:
        try:
            print("downloading", url)
            urllib.request.urlretrieve(url, DEST)
            if md5(DEST) == MD5:
                print(f"OK, verified MD5 -> {DEST}")
                return
            print("checksum mismatch, trying next mirror")
        except Exception as exc:  # network errors: try the next mirror
            print("failed:", exc)
    sys.exit("Could not fetch T1.csv - download it from Kaggle (link above) into data/raw/")


if __name__ == "__main__":
    main()
