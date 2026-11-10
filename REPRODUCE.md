# Run the HBV policy model

Code and fitted inputs for *Population impact and opportunity costs of
introducing a finite-course functional cure for chronic hepatitis B: a
global dynamic modelling study*. Download the code and input ZIP from
[the versioned release](https://github.com/Eternal2233/HBV/releases/tag/v2026.11.11-code-inputs) and extract it once. Run the commands
below from the resulting CHB_reproducibility directory.

## Restore and check the model inputs

    python3.13 prepare_inputs.py
    python3.13 -m venv ../chb-env
    source ../chb-env/bin/activate
    python -m pip install -r "代码与数据/requirements-lock.txt"
    python -m pip check
    cd "代码与数据"
    python run_final_model.py --check-inputs
    python run_key_repairs.py --check-inputs
    python run_reviewer3.py --check-inputs
    python run_feasible_repairs.py --check-inputs

The setup script verifies the standalone ZIP manifest, then restores the
pinned Seaman et al. source files from their original repository. Use
prepare_inputs.py --verify-only to verify the packaged files offline, or
--upstream-archive PATH_TO_ZIP to supply the pinned source archive separately.
On Windows use py -3.13 for setup and the environment's Scripts/Activate.ps1
in PowerShell. The native runtime documented for the study uses Python
3.13.5, Apple silicon, Xcode Command Line Tools and Apple Accelerate.

## List scenarios and run a reference case

From 代码与数据, list the 325 archived input configurations and run the
reference West Pacific Region case into a new directory:

    python run_final_model.py
    python run_final_model.py --family primary --case reference --region WPR --output ../../chb-wpr-run --kernel-cache ../../chb-native-cache

Each selected configuration runs policies A–D from the frozen inputs. Use
the case list to choose another region or scenario. Keep new calculations
and native caches outside the extracted package. The saved-output comparison
workflow uses the separate Supplementary Data 2 reviewer archive.

## Input provenance

The reporting folder contains the calibration objective, fitted coordinates,
smoothing pairs and transmission inputs. The inputs folder records the
sources, transformations, fitted parameters, costs and scenario settings.
The restore script obtains the
[original Seaman et al. source snapshot](https://github.com/ChrisSeaman-Burnet/hbv-globalinvcase/tree/c84feba26ef2f6553ab8f82ffc5ea8493474752c).
The documented WHO and HCC source references apply to the corresponding
extracted inputs. Original upstream target extraction procedures and
covariance remain unavailable.
