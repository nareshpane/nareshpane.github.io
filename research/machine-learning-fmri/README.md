# fMRI to Machine Learning Assets

This directory contains the supporting code and derived web assets for
`../machine-learning-fmri.html`. No raw participant imaging data are stored in
the repository.

## Files

- `generate_figures.py` creates `mni152_brain_views.png`,
  `schaefer_atlas_views.png`, `larger_connectivity_matrix.png`,
  `synthetic_evaluation_summary.png`, `pipeline_recap.png`, and
  `toy_connectivity.json`. The MNI152 anatomical reference is distributed with
  Nilearn. Nilearn downloads the Schaefer 2018 atlas into a cache outside this
  repository. The connectivity and evaluation values are deterministic
  synthetic teaching data.
- `abide_connectivity_baseline.py` is an optional real-data demonstration. It
  downloads a small set of ABIDE Preprocessed Connectomes Project CC200 time
  series, builds correlation features, and evaluates logistic regression with
  leave-one-site-out folds. Its printed values are not reported on the page.
- `page.js` contains only the static page's lightweight SVG/Canvas interactions.

## Reproduce The Figures

Create an isolated environment, install the packages, and run:

```bash
python -m pip install -r requirements.txt
python generate_figures.py
```

By default, `generate_figures.py` uses `/tmp/machine-learning-fmri-nilearn` for
downloaded atlas data. The ABIDE script uses `~/.cache/nilearn` unless
`NILEARN_DATA` is set. Review ABIDE's data-use terms before downloading or using
participant derivatives.

## Sources And Terms

- Nilearn: BSD 3-Clause software license, <https://nilearn.github.io/>
- Schaefer 2018 atlas: Schaefer et al., *Cerebral Cortex* 28(9),
  <https://doi.org/10.1093/cercor/bhx179>
- ABIDE: <https://fcon_1000.projects.nitrc.org/indi/abide/>
- ABIDE Preprocessed: <https://preprocessed-connectomes-project.github.io/abide/>

Software licenses do not replace the separate terms attached to downloaded
datasets and atlases. The page attributes each externally sourced numerical
resource and does not copy figures from publications.
