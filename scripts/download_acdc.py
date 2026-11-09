"""Download the ACDC cardiac MRI dataset into data/acdc/.

Run from the repo root:  uv run python scripts/download_acdc.py

Source:  https://www.creatis.insa-lyon.fr/Challenge/acdc/
License: CC BY-NC-SA 4.0 (non-commercial research). Any use must cite
Bernard et al., IEEE Transactions on Medical Imaging 37(11), 2018,
doi:10.1109/TMI.2018.2837502.
"""

import json
import urllib.request
from pathlib import Path

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
SPLITS = {
    "training": "63721d7073e9f0047faa0525",  # patients 001-100
    "testing": "6372203a73e9f0047faa117e",  # patients 101-150
}
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "acdc"


def get_json(path: str) -> list[dict]:
    """Ask the dataset server a question and return its JSON answer."""
    with urllib.request.urlopen(f"{API}/{path}", timeout=60) as response:
        return json.load(response)


def is_wanted(filename: str) -> bool:
    """Keep patient info and the ED/ES scans + masks; skip the large 4D cine files."""
    if filename == "Info.cfg":
        return True
    return filename.endswith(".nii.gz") and "_4d" not in filename


def download(item: dict, dest: Path) -> None:
    """Download one file, writing to a .part file first so a crash never leaves a half file."""
    temp = dest.parent / (dest.name + ".part")
    with urllib.request.urlopen(
        f"{API}/item/{item['_id']}/download", timeout=120
    ) as response:
        temp.write_bytes(response.read())
    if temp.stat().st_size != item["size"]:
        temp.unlink()
        raise RuntimeError(f"{dest.name}: size mismatch, try running the script again")
    temp.rename(dest)


def main() -> None:
    for split, folder_id in SPLITS.items():
        patients = get_json(
            f"folder?parentType=folder&parentId={folder_id}&limit=0&sort=name"
        )
        print(f"{split}: {len(patients)} patients")
        for number, patient in enumerate(patients, start=1):
            patient_dir = OUT_DIR / split / patient["name"]
            patient_dir.mkdir(parents=True, exist_ok=True)
            for item in get_json(f"item?folderId={patient['_id']}&limit=0"):
                dest = patient_dir / item["name"]
                already_have = dest.exists() and dest.stat().st_size == item["size"]
                if is_wanted(item["name"]) and not already_have:
                    download(item, dest)
            print(f"  [{number:3d}/{len(patients)}] {patient['name']}")
    print(f"Done. Data is in {OUT_DIR}")


if __name__ == "__main__":
    main()
