#!/usr/bin/env python3
"""
Plot CN and CA vs. Mach for each angle of attack, comparing RAS data against
DATCOM data.

One figure per alpha, with CN on the left and CA on the right. PNGs are saved
to a "plots" folder next to this script, and the figures are also shown on
screen.

Usage in VS Code:
    Put this script in the same folder as the two CSVs and press Run (F5).
    Edit RAS_FILE / DATCOM_FILE below if your files are named differently.

Requires:  pip install pandas matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---- Settings --------------------------------------------------------------
RAS_FILE = "atlas-fins-ras.csv"
DATCOM_FILE = "atlas_coefficients.csv"   # atlas_coefficients.csv also works
OUT_DIR = "plots"
SAVE_PNGS = True
SHOW_PLOTS = True
MACH_LIMITS = None                 # e.g. (0, 1.5) to zoom in on the DATCOM range
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def find(name: str) -> Path:
    """Look for a file as given, then next to this script."""
    p = Path(name)
    if p.exists():
        return p
    if (SCRIPT_DIR / name).exists():
        return SCRIPT_DIR / name
    raise SystemExit(f"File not found: {name}")


def load_ras(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values(["Alpha", "Mach"])


def load_datcom(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept either the raw (mach, ALPHA) or reshaped (Mach, Alpha) naming
    df = df.rename(columns={"mach": "Mach", "ALPHA": "Alpha"})
    return df.sort_values(["Alpha", "Mach"])


def plot_alpha(alpha: float, ras: pd.DataFrame, datcom: pd.DataFrame):
    r = ras[ras["Alpha"] == alpha]
    d = datcom[datcom["Alpha"] == alpha]

    fig, (ax_cn, ax_ca) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    for ax, coef in ((ax_cn, "CN"), (ax_ca, "CA")):
        if not r.empty:
            ax.plot(r["Mach"], r[f"{coef} Power-Off"], color="tab:blue",
                    label="RAS power-off")
            ax.plot(r["Mach"], r[f"{coef} Power-On"], color="tab:blue",
                    linestyle="--", label="RAS power-on")
        if not d.empty:
            # NaNs (DATCOM gaps) are left in on purpose so the line breaks
            # there instead of drawing across the missing region.
            ax.plot(d["Mach"], d[coef], color="tab:red", marker="o",
                    markersize=4, label="DATCOM")
        ax.set_xlabel("Mach")
        ax.set_ylabel(coef)
        ax.set_title(f"{coef} vs. Mach")
        ax.grid(True, alpha=0.3)
        if MACH_LIMITS:
            ax.set_xlim(*MACH_LIMITS)

    ax_cn.legend()
    fig.suptitle(f"Alpha = {alpha:g}\N{DEGREE SIGN}")
    fig.tight_layout()
    return fig


def main() -> None:
    ras = load_ras(find(RAS_FILE))
    datcom = load_datcom(find(DATCOM_FILE))

    alphas = sorted(set(ras["Alpha"]) | set(datcom["Alpha"]))

    out_dir = SCRIPT_DIR / OUT_DIR
    if SAVE_PNGS:
        out_dir.mkdir(exist_ok=True)

    for alpha in alphas:
        fig = plot_alpha(alpha, ras, datcom)
        if SAVE_PNGS:
            fig.savefig(out_dir / f"cn_ca_alpha_{alpha:g}.png", dpi=150)

    print(f"Made {len(alphas)} figures for alpha = {[float(a) for a in alphas]}")
    if SAVE_PNGS:
        print(f"Saved PNGs to {out_dir}")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
