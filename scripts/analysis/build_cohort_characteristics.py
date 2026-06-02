"""Build the Table 1 cohort-characteristics summary for the PI-CAI 1500 cohort.

Compares csPCa-positive (ISUP >= 2) versus csPCa-negative cases on the available
clinical variables, using non-parametric tests (continuous variables are skewed),
and writes CSV, Markdown, and LaTeX versions ready to drop into the manuscript.

Usage
-----
python scripts/analysis/build_cohort_characteristics.py \
    --data artifacts/data.csv \
    --output-dir results/radiomics/picai1500_corr/publication_report/cohort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS = [
    ("patient_age", "Age, years"),
    ("psa", "PSA, ng/mL"),
    ("psad", "PSA density, ng/mL/mL"),
    ("prostate_volume", "Prostate volume, mL"),
]


def med_iqr(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "--"
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/data.csv")
    parser.add_argument(
        "--output-dir",
        default="results/radiomics/picai1500_corr/publication_report/cohort",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.data)
    d["case_csPCa"] = d["case_csPCa"].astype(int)
    pos = d[d.case_csPCa == 1]
    neg = d[d.case_csPCa == 0]

    rows = []

    # Header counts.
    rows.append(
        {
            "Variable": "Cases, n",
            "Overall": f"{len(d)}",
            "Non-csPCa": f"{len(neg)}",
            "csPCa": f"{len(pos)}",
            "Missing, n": "0",
            "p-value": "",
            "Test": "",
        }
    )
    rows.append(
        {
            "Variable": "Patients, n",
            "Overall": f"{d.patient_id.nunique()}",
            "Non-csPCa": f"{neg.patient_id.nunique()}",
            "csPCa": f"{pos.patient_id.nunique()}",
            "Missing, n": "0",
            "p-value": "",
            "Test": "",
        }
    )

    # Continuous variables: median (IQR), Mann-Whitney U.
    for col, label in CONTINUOUS:
        a = pos[col].dropna()
        b = neg[col].dropna()
        if len(a) >= 2 and len(b) >= 2:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            p_str = f"{p:.3g}"
        else:
            p_str = "--"
        rows.append(
            {
                "Variable": f"{label}, median (IQR)",
                "Overall": med_iqr(d[col]),
                "Non-csPCa": med_iqr(neg[col]),
                "csPCa": med_iqr(pos[col]),
                "Missing, n": f"{int(d[col].isna().sum())}",
                "p-value": p_str,
                "Test": "Mann-Whitney U",
            }
        )

    # ISUP grade group distribution.
    isup_counts = d["case_ISUP"].value_counts().sort_index()
    for grade, n in isup_counts.items():
        n_pos = int(((d.case_ISUP == grade) & (d.case_csPCa == 1)).sum())
        n_neg = int(((d.case_ISUP == grade) & (d.case_csPCa == 0)).sum())
        rows.append(
            {
                "Variable": f"ISUP grade group {int(grade)}, n (%)",
                "Overall": f"{int(n)} ({100 * n / len(d):.1f})",
                "Non-csPCa": f"{n_neg}" if n_neg else "0",
                "csPCa": f"{n_pos}" if n_pos else "0",
                "Missing, n": "0",
                "p-value": "",
                "Test": "",
            }
        )

    table = pd.DataFrame(rows)

    csv_path = out_dir / "cohort_characteristics.csv"
    table.to_csv(csv_path, index=False)

    # Markdown.
    md = ["# Table 1. Cohort characteristics (PI-CAI 1500)", ""]
    md.append("csPCa is defined as ISUP grade group >= 2.")
    md.append("")
    md.append("| " + " | ".join(table.columns) + " |")
    md.append("|" + "|".join(["---"] * len(table.columns)) + "|")
    for _, r in table.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in table.columns) + " |")
    md.append("")
    (out_dir / "cohort_characteristics.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX (booktabs).
    def esc(x: str) -> str:
        return str(x).replace("%", r"\%").replace("_", r"\_")

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cohort characteristics for the PI-CAI 1500 cohort. "
        r"Clinically significant prostate cancer (csPCa) is defined as ISUP grade "
        r"group $\geq 2$. Continuous variables are summarised as median "
        r"(interquartile range) and compared with the Mann--Whitney U test.}",
        r"\label{tab:cohort}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variable & Overall & Non-csPCa & csPCa & Missing, n & $p$-value \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        tex.append(
            " & ".join(
                [
                    esc(r["Variable"]),
                    esc(r["Overall"]),
                    esc(r["Non-csPCa"]),
                    esc(r["csPCa"]),
                    esc(r["Missing, n"]),
                    esc(r["p-value"]),
                ]
            )
            + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "cohort_characteristics.tex").write_text("\n".join(tex), encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {out_dir / 'cohort_characteristics.md'}")
    print(f"Wrote: {out_dir / 'cohort_characteristics.tex'}")


if __name__ == "__main__":
    main()
