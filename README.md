# Hepatitis B policy modelling

Reproducibility materials for *Population impact and opportunity costs of introducing a finite-course functional cure for chronic hepatitis B: a global dynamic modelling study*.

## November 2026 reproducibility release

Download the two assets from the [versioned release](https://github.com/Eternal2233/HBV/releases/tag/v2026.11.10-reproducibility). They contain the analysis code, fitted inputs and saved results for the six WHO regions. The repository tree also preserves the earlier model files.

| Asset | Contents |
| --- | --- |
| [Code and inputs](https://github.com/Eternal2233/HBV/releases/download/v2026.11.10-reproducibility/CHB-code-inputs-v2026.11.10-reproducibility.zip) | Analysis scripts, model workbooks, fitted parameters, scenario specifications, dependency lock and source checksums |
| [Saved results](https://github.com/Eternal2233/HBV/releases/download/v2026.11.10-reproducibility/CHB-saved-results-v2026.11.10-reproducibility.zip) | Annual model arrays, conditional PSA draws, budget checks, table data and figure data |

Extract both into the same location and open [REPRODUCE.md](REPRODUCE.md). The quick verification rebuilds 1,300 policy totals and the summaries for all 3,500 saved PSA draws. The model replay instructions run a four-policy reference case from the fitted inputs and compare the annual arrays. The guide also provides the commands for the other regions, sensitivities and figure generation.

The native model replay was tested with Python 3.13.5 on Apple silicon. The PSA is conditional on the reference fits and model structure. Computational reproduction checks agreement under those inputs and assumptions.
