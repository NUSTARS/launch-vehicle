#!/usr/bin/env python3
"""
Convert one or more aerodynamic coefficient .txt outputs (Mach/Alpha sweep,
block format) into a single, flat, well-organized .csv file.

Each input .txt typically corresponds to a single angle of attack (AoA) swept
across Mach numbers; multiple files (one per AoA) can be passed in and their
rows are stacked on top of each other in the output, in the order given.

Expected .txt structure
------------------------
Blocks separated by one or more blank lines. Each block corresponds to one
Mach number and looks like:

  Line 1 (block header, field count VARIES by flight regime):
      Mach, Alpha(=0), CDPowerOff, CDPowerOn, [...drag breakdown terms whose
      count/labels differ by regime -- e.g. subsonic includes Body/Fin
      Frict/Press/Base terms; transonic has none; supersonic/hypersonic adds
      wave-drag terms...], ReynoldsNo

      Only two positions from this line are used, since they're the only
      ones constant across all regimes: the LAST field (ReynoldsNo).
      Everything else in this line is regime-specific and dropped.

  Line 2 (constant for the whole block, 3 values):
      Mach, CNAlpha (0-4 deg), CP (0-4 deg)

  Then, repeating in groups of 4 lines -- one group per Alpha value present
  in the block (e.g. Alpha = 0, 2, 4):
      Mach, Alpha, CN Potential, CN Viscous
      Mach, Alpha, CN Total, CP Total
      Mach, Alpha, CL Power Off, CD Power Off, CN Power Off, CA Power Off
      Mach, Alpha, CL Power On,  CD Power On,  CN Power On,  CA Power On

Output columns (one row per Mach/Alpha combination, Mach <= MAX_MACH only)
---------------------------------------------------------------------------
Mach, Alpha, CD Power-Off, CD Power-On, CA Power-Off, CA Power-On,
CL Power-Off, CL Power-On, CN Power-Off, CN Power-On,
CNalpha (0 to 4 deg) (per rad), CP, Reynolds Number

Note: CP is taken from the "CP Total" field (not the "CP (0-4 deg)" field).
"""

import sys
from pathlib import Path

import pandas as pd

MAX_MACH = 5.0  # rows with Mach greater than this are dropped from the output

OUTPUT_HEADER = [
    "Mach",
    "Alpha",
    "CD Power-Off",
    "CD Power-On",
    "CA Power-Off",
    "CA Power-On",
    "CL Power-Off",
    "CL Power-On",
    "CN Power-Off",
    "CN Power-On",
    "CNalpha (0 to 4 deg) (per rad)",
    "CP",
    "Reynolds Number",
]


def read_blocks(path):
    """Yield lists of non-empty, whitespace-split lines, one list per block."""
    block = []
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if block:
                    yield block
                    block = []
                continue
            block.append(line.split())
    if block:
        yield block


def parse_block(block):
    """Parse a single Mach-block into a list of output rows."""
    rows = []

    if len(block) < 2:
        return rows  # malformed/too-short block, skip

    header = block[0]
    cn_line = block[1]

    # --- Block header: Mach, Alpha, CDPowerOff, CDPowerOn, [...regime-specific
    #     drag breakdown terms...], ReynoldsNo
    #
    #     Regardless of regime, the LAST field is always Reynolds No, so key
    #     off the end rather than a fixed field count.
    if len(header) < 5:
        return rows
    reynolds = float(header[-1])

    # --- CNalpha / CP (0-4 deg), constant for the whole block
    if len(cn_line) < 3:
        return rows
    cn_alpha = float(cn_line[1])
    # cn_line[2] is CP (0-4 deg) -- not used in the output (CP comes from
    # the "CP Total" field further down instead).

    # --- Remaining lines: groups of 4, one group per Alpha value
    remaining = block[2:]
    for i in range(0, len(remaining) - 3, 4):
        potvisc, total, poff, pon = remaining[i:i + 4]

        if len(potvisc) < 4 or len(total) < 4 or len(poff) < 6 or len(pon) < 6:
            continue  # incomplete group, skip

        mach = float(potvisc[0])
        alpha = float(potvisc[1])

        cp_total = float(total[3])

        cl_poff = float(poff[2])
        cd_poff = float(poff[3])
        cn_poff = float(poff[4])
        ca_poff = float(poff[5])

        cl_pon = float(pon[2])
        cd_pon = float(pon[3])
        cn_pon = float(pon[4])
        ca_pon = float(pon[5])

        rows.append([
            mach,
            alpha,
            cd_poff,
            cd_pon,
            ca_poff,
            ca_pon,
            cl_poff,
            cl_pon,
            cn_poff,
            cn_pon,
            cn_alpha,
            cp_total,
            reynolds,
        ])

    return rows


def convert(input_paths, output_path):
    """Parse one or more .txt files and stack all resulting rows into one CSV."""
    all_rows = []
    for input_path in input_paths:
        for block in read_blocks(input_path):
            all_rows.extend(parse_block(block))

    df = pd.DataFrame(all_rows, columns=OUTPUT_HEADER)
    df = df[df["Mach"] <= MAX_MACH].reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} rows (Mach <= {MAX_MACH}) to {output_path}")


if __name__ == "__main__":
    # --- Option A: run from the command line ---
    #     python3 convert_aero_txt_to_csv.py <output.csv> <input1.txt> [input2.txt ...]
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[1])
        in_paths = [Path(p) for p in sys.argv[2:]]
        convert(in_paths, out_path)
    else:
        # --- Option B: no arguments given (e.g. hitting Run/F5 in VS Code)
        #     -> edit the paths below and just click Run.
        in_paths = [
            Path("analysis/aero/raw-data/atlas-nofins-0.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-1.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-2.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-3.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-4.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-5.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-6.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-7.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-8.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-9.txt"),
            Path("analysis/aero/raw-data/atlas-nofins-10.txt"),
        ]
        out_path = Path("analysis/aero/atlas-nofins-ras.csv")
        convert(in_paths, out_path)