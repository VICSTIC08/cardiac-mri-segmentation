"""Load ACDC patients: patient info (Info.cfg) plus the ED and ES scans and masks."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

LABELS = {0: "background", 1: "RV", 2: "myocardium", 3: "LV"}


@dataclass
class Scan:
    """One 3D scan (all slices at one moment of the heartbeat) and its mask."""

    image: np.ndarray  # float32, shape (width, height, slices)
    mask: np.ndarray  # uint8, same shape, labels 0-3
    spacing: tuple[float, float, float]  # voxel size in mm: (width, height, depth)


@dataclass
class Patient:
    """Everything we use from one ACDC patient folder."""

    patient_id: str  # e.g. "patient001"
    group: str  # diagnosis group, e.g. "DCM"
    ed: Scan  # end-diastole: heart fullest
    es: Scan  # end-systole: heart most squeezed


def read_info(patient_dir: Path) -> dict[str, str]:
    """Read Info.cfg into a dictionary, e.g. {"ED": "1", "ES": "12", "Group": "DCM", ...}."""
    info = {}
    for line in (patient_dir / "Info.cfg").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def load_scan(patient_dir: Path, frame: int) -> Scan:
    """Load one frame's scan and its doctors' mask, and check they match."""
    stem = f"{patient_dir.name}_frame{frame:02d}"
    image_file = nib.load(patient_dir / f"{stem}.nii.gz")
    mask_file = nib.load(patient_dir / f"{stem}_gt.nii.gz")

    image = image_file.get_fdata(dtype=np.float32)
    mask = np.rint(mask_file.get_fdata()).astype(np.uint8)
    spacing = tuple(float(size) for size in image_file.header.get_zooms()[:3])

    if image.shape != mask.shape:
        raise ValueError(f"{stem}: image {image.shape} and mask {mask.shape} differ")
    return Scan(image=image, mask=mask, spacing=spacing)


def load_patient(patient_dir: Path) -> Patient:
    """Load a whole patient: info, ED scan + mask, ES scan + mask."""
    patient_dir = Path(patient_dir)
    info = read_info(patient_dir)
    return Patient(
        patient_id=patient_dir.name,
        group=info["Group"],
        ed=load_scan(patient_dir, int(info["ED"])),
        es=load_scan(patient_dir, int(info["ES"])),
    )


def list_patients(split_dir: Path) -> list[Path]:
    """All patient folders in data/acdc/training or data/acdc/testing, in order."""
    return sorted(
        path
        for path in Path(split_dir).iterdir()
        if path.is_dir() and path.name.startswith("patient")
    )
