#!/usr/bin/env python3
"""
Convert aerodynamic coefficient .txt output (Mach/Alpha sweep, block format)
into a flat, well-organized .csv file.

Expected .txt structure
------------------------
Blocks separated by one or more blank lines. Each block corresponds to one
Mach number and looks like:

  Line 1 (block header, field count VARIES by flight regime):
      Mach, Alpha(=0), CDPowerOff, CDPowerOn, [...drag breakdown terms whose
      count/labels differ by regime -- e.g. subsonic includes Body/Fin
      Frict/Press/Base terms; transonic has none; supersonic/hypersonic adds
      wave-drag terms...], ReynoldsNo

      Only two positions are used here since they're the only ones constant
      across all regimes: index 2 (CDPowerOff) and the LAST field
      (ReynoldsNo). Everything else in this line is regime-specific and
      dropped.

  Line 2 (constant for the whole block, 3 values):
      Mach, CNAlpha (0-4 deg), CP (0-4 deg)

  Then, repeating in groups of 4 lines -- one group per Alpha value present
  in the block (e.g. Alpha = 0, 2, 4):
      Mach, Alpha, CN Potential, CN Viscous
      Mach, Alpha, CN Total, CP Total
      Mach, Alpha, CL Power Off, CD Power Off, CN Power Off, CA Power Off
      Mach, Alpha, CL Power On,  CD Power On,  CN Power On,  CA Power On

Output columns (one row per Mach/Alpha combination)
----------------------------------------------------
Mach, Alpha, CD, CD Power-Off, CD Power-On, CA Power-Off, CA Power-On, CL,
CN, CN Potential, CN Viscous, CNalpha (0 to 4 deg) (per rad), CP,
CP (0 to 4 deg), Reynolds Number
"""

import sys
from pathlib import Path

import pandas as pd

OUTPUT_HEADER = [
    "Mach",
    "Alpha",
    "CD",
    "CD Power-Off",
    "CD Power-On",
    "CA Power-Off",
    "CA Power-On",
    "CL",
    "CN",
    "CN Potential",
    "CN Viscous",
    "CNalpha (0 to 4 deg) (per rad)",
    "CP",
    "CP (0 to 4 deg)",
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
    #     drag breakdown terms, count varies: subsonic has body/fin breakdown,
    #     transonic has none, supersonic/hypersonic has wave-drag terms...],
    #     ReynoldsNo
    #
    #     Regardless of regime, field 2 (index 2) is always CD Power-Off and
    #     the LAST field is always Reynolds No, so key off position/end
    #     rather than a fixed field count.
    if len(header) < 5:
        return rows
    cd_header = float(header[2])       # constant "CD" for this block
    reynolds = float(header[-1])

    # --- CNalpha / CP (0-4 deg), constant for the whole block
    if len(cn_line) < 3:
        return rows
    cn_alpha = float(cn_line[1])
    cp_04 = float(cn_line[2])

    # --- Remaining lines: groups of 4, one group per Alpha value
    remaining = block[2:]
    for i in range(0, len(remaining) - 3, 4):
        potvisc, total, poff, pon = remaining[i:i + 4]

        if len(potvisc) < 4 or len(total) < 4 or len(poff) < 6 or len(pon) < 6:
            continue  # incomplete group, skip

        mach = float(potvisc[0])
        alpha = float(potvisc[1])
        cn_potential = float(potvisc[2])
        cn_viscous = float(potvisc[3])

        cn_total = float(total[2])
        cp_total = float(total[3])

        cl = float(poff[2])           # CL is the same for power-off/on
        cd_poff = float(poff[3])
        ca_poff = float(poff[5])

        cd_pon = float(pon[3])
        ca_pon = float(pon[5])

        rows.append([
            mach,
            alpha,
            cd_header,
            cd_poff,
            cd_pon,
            ca_poff,
            ca_pon,
            cl,
            cn_total,
            cn_potential,
            cn_viscous,
            cn_alpha,
            cp_total,
            cp_04,
            reynolds,
        ])

    return rows


def convert(input_path, output_path):
    all_rows = []
    for block in read_blocks(input_path):
        all_rows.extend(parse_block(block))

    df = pd.DataFrame(all_rows, columns=OUTPUT_HEADER)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python convert_aero_txt_to_csv.py <input.txt> <output.csv>")
    #     sys.exit(1)

    # in_path = Path(sys.argv[1])
    # out_path = Path(sys.argv[2])
    in_path = 'analysis/aero/raw-data/atlas-fins-2.txt'
    out_path = 'analysis/aero/atlas-fins-2.csv'
    convert(in_path, out_path)