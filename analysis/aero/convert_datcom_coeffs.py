#!/usr/bin/env python3
"""
Reshape an atlas_coefficients-style CSV into the atlas-fins-ras layout:
Mach in the leftmost column, Alpha second, and rows ordered so that every
Mach number is swept at one angle of attack before moving to the next alpha.

All original columns are kept (after Mach and Alpha), and blank cells stay blank.

Usage in VS Code:
    Put this script in the same folder as atlas_coefficients.csv, open it,
    and press the Run button (or F5). It writes atlas_sorted.csv next to it.
    To use different files, edit INPUT_FILE / OUTPUT_FILE below.

Usage from a terminal (optional):
    python convert_atlas.py [input.csv] [output.csv]

Requires pandas:  pip install pandas
"""
import argparse
from pathlib import Path

import pandas as pd

# Edit these if your files are named differently. Relative names are looked
# up in the same folder as this script, no matter where VS Code's terminal is.
INPUT_FILE = "atlas_coefficients.csv"
OUTPUT_FILE = "atlas_datcom_coeffs.csv"
SCRIPT_DIR = Path(__file__).resolve().parent


def convert(in_path: str, out_path: str) -> pd.DataFrame:
    df = pd.read_csv(in_path)

    # Match the capitalization used in atlas-fins-ras
    df = df.rename(columns={"mach": "Mach", "ALPHA": "Alpha"})

    # Mach first, Alpha second, everything else after in original order
    front = ["Mach", "Alpha"]
    df = df[front + [c for c in df.columns if c not in front]]

    # Alpha is the slow (outer) index, Mach the fast (inner) one
    df = df.sort_values(["Alpha", "Mach"], kind="stable").reset_index(drop=True)

    df.to_csv(out_path, index=False, lineterminator="\r\n")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=INPUT_FILE,
                        help=f"atlas_coefficients-style CSV (default: {INPUT_FILE})")
    parser.add_argument("output", nargs="?", default=OUTPUT_FILE,
                        help=f"path for the reshaped CSV (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_absolute() and not in_path.exists():
        in_path = SCRIPT_DIR / in_path
    if not out_path.is_absolute() and args.output == OUTPUT_FILE:
        out_path = SCRIPT_DIR / out_path
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = convert(str(in_path), str(out_path))
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {out_path}")
    print(f"Alphas: {[float(a) for a in sorted(df['Alpha'].unique())]}")
    print(f"Mach numbers per alpha: {df['Mach'].nunique()}")


if __name__ == "__main__":
    main()
