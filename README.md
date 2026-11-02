# CloudCharge-LayerCloud

**Charge-resolved transient electrification model for layer-cloud boundaries**

This GitHub repository contains the source code, configurations, tests,
documentation, and lightweight manuscript-level tables supporting the JGR:
Atmospheres manuscript:

> **Charge-Resolved Transient Electrification at Layer-Cloud Boundaries**

Large primary model outputs and 300 dpi publication figures are intentionally
archived separately in a Zenodo Dataset. This keeps the GitHub repository
small, reviewable, and suitable for long-term software development.

## Version

- CloudCharge software: **5.0.0**
- Paper-specific data archive: **1.0**

## Structure

```text
CloudCharge-LayerCloud/
├── README.md
├── LICENSE
├── DATA_LICENSE.md
├── CITATION.cff
├── .gitignore
├── REPRODUCIBILITY.md
├── requirements.txt
├── RELEASE_NOTES_v5.0.0.md
├── model/
├── data/
│   ├── manuscript_tables/
│   └── README.md
├── figures/
│   └── README.md
└── docs/
    ├── AGU_OPEN_RESEARCH.md
    ├── ZENODO_DATA_UPLOAD.md
    └── ZENODO_RELEASE_CHECKLIST.md
```

## Installation

```bash
git clone https://github.com/zhoulim-cell/CloudCharge_Paper_Reproducibility_Package.git
cd CloudCharge_Paper_Reproducibility_Package

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Validation

```bash
cd model
python check_environment.py
python validate_papers.py
```

Additional validation and regression tests are available under `model/tests/`.

## Data availability

Large numerical outputs and final publication figures are deposited separately
in a Zenodo Dataset.

**Dataset DOI:** `https://doi.org/10.5281/zenodo.DATA_DOI`

After Zenodo assigns the real dataset DOI, replace the placeholder above and
the matching placeholders in `docs/AGU_OPEN_RESEARCH.md`.

## Citation

CloudCharge v5.0.0 is archived in Zenodo and has the following
version-specific software DOI:

**Software DOI:** https://doi.org/10.5281/zenodo.23879986

Please cite the Zenodo software record rather than only the mutable GitHub
`main` branch.

## License

Source code is released under the MIT License. Paper-supporting research data
are released separately under CC BY 4.0.
