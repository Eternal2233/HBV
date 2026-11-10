# Reproducing the HBV policy modelling results

This is the November 2026 analysis snapshot for *Population impact and opportunity costs of introducing a finite-course functional cure for chronic hepatitis B: a global dynamic modelling study*. Cite the [versioned GitHub release](https://github.com/Eternal2233/HBV/releases/tag/v2026.11.10-reproducibility) when reporting a reproduction.

## Set up the release

1. Download both `CHB-code-inputs-v2026.11.10-reproducibility.zip` and `CHB-saved-results-v2026.11.10-reproducibility.zip` from the release assets.
2. Extract both ZIPs into the same directory. They form one `CHB_reproducibility` folder and share an identical `SHA256SUMS.txt`. If your extractor asks about that one file, keep either copy.
3. Use Python 3.13.5. From inside `CHB_reproducibility`, run:

```bash
python3.13 prepare_inputs.py
python3.13 -m venv ../chb-env
source ../chb-env/bin/activate
python -m pip install -r "代码与数据/requirements-lock.txt"
python -m pip check
```

`prepare_inputs.py` verifies all bundled files and restores each pinned upstream input from its publisher with a SHA-256 check. For an offline check of the two release archives use `python3.13 prepare_inputs.py --verify-only`. For offline restoration, download the [pinned upstream ZIP](https://codeload.github.com/ChrisSeaman-Burnet/hbv-globalinvcase/zip/c84feba26ef2f6553ab8f82ffc5ea8493474752c) separately and pass `--upstream-archive PATH_TO_ZIP`. On Windows, use `py -3.13` and activate `..\chb-env\Scripts\Activate.ps1` in PowerShell. The environment and saved-result commands were tested on macOS arm64. The saved-result workflow requires several minutes and approximately 2 GB of free disk space for archives, dependencies and output.

## Recompute the reported results from saved arrays

```bash
python verify_saved_results.py --output ../chb-results-check
```

The script rebuilds and compares the 1,300 policy totals from 325 regional scenarios. It recomputes the conditional PSA summaries from the 3,500 saved joint draws, checks the 21,000 draw-by-region budget records, and rebuilds the 108 regional joint sensitivity comparisons. New reports are written outside the release folder. The comparison uses relative and absolute tolerances of `1e-11` and `1e-7` for exported numeric values. This workflow uses saved results and performs zero new simulations.

- **Table data.** `输出结果/统计表/发表补充表/publication_tables.json` contains Supplementary Tables S1–S38. `输出结果/Source_Data_Index.txt` maps the result families to their source files.
- **PSA data.** `输出结果/关键问题专项/joint_psa_summary/` contains the draw-level results, intervals, event frequencies and budget checks. The six regional records for a given draw ID share one joint parameter vector.
- **Figures.** With Times New Roman regular, bold, italic and bold italic installed, run the following from the release root to regenerate 16 figure sets from their source tables.

```bash
python "代码与数据/redraw_final_figures.py" --output ../chb-figures
```

## Run the model from the frozen inputs

The reference native environment uses Python 3.13.5 on Apple silicon with Xcode Command Line Tools and Apple Accelerate. The first run builds the native kernel and can take several minutes. From the release root:

```bash
cd "代码与数据"
python run_final_model.py --check-inputs
python run_key_repairs.py --check-inputs
python run_final_model.py --family primary --case reference --region WPR \
  --output ../../chb-wpr-replay --kernel-cache ../../chb-native-cache
```

The runner rematches the B/C budget and recomputes policies A–D. Verify its annual arrays from the same directory:

```bash
python - <<'PY'
import numpy as np

reference = "../输出结果/年度模型结果/primary/reference/WPR/arrays.npz"
replay = "../../chb-wpr-replay/arrays.npz"
with np.load(reference, allow_pickle=False) as saved, np.load(replay, allow_pickle=False) as new:
    assert set(saved.files) == set(new.files)
    for key in saved.files:
        np.testing.assert_allclose(saved[key], new[key], rtol=1e-7, atol=1e-4, err_msg=key)
    print(f"{len(saved.files)} annual arrays match.")
PY
```

Run `python run_final_model.py` to list the remaining indexed cases. For the full six-region PSA, the joint stress grid, and the other analyses, follow `代码与数据/README_REPRODUCIBILITY.txt`. Use a separate output directory per independent workflow.

## Source provenance

The release includes aggregate analysis inputs, fitted parameters, scenario specifications and saved results. The archive in `upstream_sources.json` provides the original Seaman et al. model inputs. The release supplies the extracted HCC and clinic-price tables that were used in the analyses. Use the publisher's supplemental Table 7 and the WHO report pages 56–60 to check their upstream values. The report URL and SHA-256 are recorded in `WHO_CHOICE_cost_audit.json`. Local machine paths in historical provenance files have been redacted. The scientific settings and all saved arrays and numeric CSVs retain their archived values. [Seaman et al. source snapshot](https://github.com/ChrisSeaman-Burnet/hbv-globalinvcase/tree/c84feba26ef2f6553ab8f82ffc5ea8493474752c), [WHO-CHOICE cost estimates](https://www.who.int/publications/m/item/who-choice-estimates-of-cost-for-inpatient-and-outpatient-health-service-delivery), [Cao et al. HCC estimates](https://doi.org/10.1136/gutjnl-2025-337838).

WHO Global Health Estimates extracts cover selected hepatitis B outcomes by region, age and sex. WHO and UNICEF vaccination estimates describe country coverage heterogeneity. The release records source URLs, retrieval dates and transformations alongside the extracts. [WHO data terms](https://www.who.int/about/policies/publishing/data-policy/terms-and-conditions), [WHO/UNICEF hepatitis B vaccination estimates](https://immunizationdata.who.int/global/wiise-detail-page/hepatitis-b-vaccination-coverage).

Computational reproduction establishes numerical agreement under the frozen inputs and assumptions. The PSA varies the specified clinical and cost inputs conditional on the reference fits and structure. The original upstream target extraction and covariance remain unavailable, and independent predictive validity remains unestablished.
