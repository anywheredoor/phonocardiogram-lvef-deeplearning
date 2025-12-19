#!/usr/bin/env python3
"""
Build a metadata table linking each PCG .wav file to:
- patient_id
- EF value
- binary EF label (1 if EF <= 40, else 0)
- device (parsed from filename)
- auscultation position (parsed from filename)
- relative file path

Run from the repository root, e.g.:

    python -m src.data.build_metadata \
        --lvef_csv lvef.csv \
        --heart_dir heart_sounds \
        --output_csv metadata.csv
"""

import argparse
import os
import re
import sys
from typing import Dict, List

import pandas as pd

# -----------------------------------------------------------------------------
# Assumptions about filename pattern and device codes
# -----------------------------------------------------------------------------
# Assumed filename pattern, e.g. "aData2001A.wav":
#   <device_code>Data<patient_id><position>.wav
#
# Examples:
#   aData2001A.wav
#   eData2573T.wav
#   iData2100M.wav
#
# If your naming convention is different, adjust the regex below.
FILENAME_RE = re.compile(
    r'^(?P<device_code>[A-Za-z])Data(?P<patient_id>\d+)(?P<position>[A-Za-z])\.wav$',
    flags=re.IGNORECASE,
)

# Map one-letter device codes to human-readable device names.
# NOTE: Adjust if your convention is different.
DEVICE_MAP = {
    "a": "android_phone",
    "i": "iphone",
    "e": "digital_stethoscope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata.csv linking PCG files to EF labels."
    )
    parser.add_argument(
        "--lvef_csv",
        type=str,
        default="lvef.csv",
        help="Path to lvef CSV file (default: lvef.csv)",
    )
    parser.add_argument(
        "--heart_dir",
        type=str,
        default="heart_sounds",
        help="Root directory containing per-patient subfolders (default: heart_sounds)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="metadata.csv",
        help="Output metadata CSV path (default: metadata.csv)",
    )
    return parser.parse_args()


def load_lvef_table(lvef_csv: str) -> pd.DataFrame:
    if not os.path.isfile(lvef_csv):
        print(f"ERROR: lvef CSV not found at {lvef_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(lvef_csv)

    required_cols = {"patient_id", "ef"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"ERROR: lvef CSV is missing required columns: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Drop missing EF values, as agreed
    before = len(df)
    df = df.dropna(subset=["ef"]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing EF values.")

    # Ensure patient_id is string so it matches directory names
    df["patient_id"] = df["patient_id"].astype(str)

    return df


def build_metadata(
    df_lvef: pd.DataFrame, heart_dir: str
) -> pd.DataFrame:
    # Map patient_id -> ef
    ef_map: Dict[str, float] = (
        df_lvef.set_index("patient_id")["ef"].astype(float).to_dict()
    )

    rows: List[dict] = []

    n_patients_with_dir = 0
    n_patients_without_dir = 0

    for pid, ef in ef_map.items():
        patient_dir = os.path.join(heart_dir, pid)
        if not os.path.isdir(patient_dir):
            # Not necessarily an error: maybe some EF entries have no recordings.
            print(
                f"Warning: no heart_sounds directory found for patient_id={pid} "
                f"at {patient_dir}. Skipping this patient."
            )
            n_patients_without_dir += 1
            continue

        n_patients_with_dir += 1

        for fname in os.listdir(patient_dir):
            if not fname.lower().endswith(".wav"):
                continue  # Ignore non-wav files

            m = FILENAME_RE.match(fname)
            if not m:
                print(
                    f"Warning: filename '{fname}' in {patient_dir} does not "
                    f"match expected pattern. Skipping this file."
                )
                continue

            fname_pid = m.group("patient_id")
            device_code = m.group("device_code").lower()
            position = m.group("position").upper()

            # Sanity check: filename patient_id vs directory name
            if fname_pid != pid:
                print(
                    f"Warning: patient_id mismatch for file '{fname}': "
                    f"directory={pid}, filename={fname_pid}"
                )

            device = DEVICE_MAP.get(device_code, "unknown")

            rel_path = os.path.join(heart_dir, pid, fname)

            rows.append(
                {
                    "patient_id": int(pid),  # keep as int in output for convenience
                    "ef": float(ef),
                    "label": int(float(ef) <= 40.0),  # 1 if EF <= 40 else 0
                    "device_code": device_code,
                    "device": device,
                    "position": position,
                    "filename": fname,
                    "path": rel_path,
                }
            )

    if not rows:
        print(
            "ERROR: No .wav files were found or parsed successfully. "
            "Check your heart_sounds directory and filename pattern.",
            file=sys.stderr,
        )
        sys.exit(1)

    df_meta = pd.DataFrame(rows)

    # Sort for reproducibility: by patient_id, then device, then position, then filename
    df_meta = df_meta.sort_values(
        by=["patient_id", "device_code", "position", "filename"]
    ).reset_index(drop=True)

    print(
        f"Built metadata for {df_meta['patient_id'].nunique()} patients "
        f"and {len(df_meta)} recordings."
    )
    print(
        f"Patients with recordings: {n_patients_with_dir}, "
        f"patients missing recordings: {n_patients_without_dir}"
    )

    return df_meta


def main():
    args = parse_args()

    df_lvef = load_lvef_table(args.lvef_csv)
    df_meta = build_metadata(df_lvef, args.heart_dir)

    out_path = args.output_csv
    df_meta.to_csv(out_path, index=False)
    print(f"Metadata saved to: {out_path}")


if __name__ == "__main__":
    main()
