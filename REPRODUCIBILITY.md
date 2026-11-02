# Reproducibility Guide

This GitHub repository contains the executable software, tests, configurations,
documentation, and lightweight final manuscript tables.

Large raw model outputs and publication TIFF files are not tracked in GitHub;
they are archived in the paper-supporting Zenodo Dataset (version 1.0).

## Baseline checks

```bash
cd model
python check_environment.py
python validate_papers.py
```

## Paper-supporting data

Exact manuscript-level numeric tables are under:

```text
data/manuscript_tables/
```

The separate Zenodo Dataset contains the larger primary outputs used for the
expanded-domain and ZT07 analyses together with figure files and integrity
manifests.

## Completeness note

Some compact-domain sensitivity/convergence experiments are represented by
exact manuscript-level summary tables even when their individual raw run
directories were not preserved in the original working archive. This
limitation should remain explicit in the JGR Open Research statement.
