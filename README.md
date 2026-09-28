# Predictive Manpower Planning for Technical Support

Machine learning project exploring how technical-support ticket data can support staffing decisions, resource allocation, product expertise planning, and service quality.

## GitHub Pages deployment

This repository is designed to be published as a **project site** under the GitHub account `Herteln`.

Expected URL:

`https://herteln.github.io/Predictive-Manpower-Planning/`

### Publish

1. Upload all files and folders from this package to the root of the `Predictive-Manpower-Planning` repository.
2. Make sure `index.html` is directly in the repository root (not inside another folder).
3. In GitHub open **Settings > Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select branch **main** and folder **/(root)**.
6. Click **Save**.
7. Wait for GitHub Pages to finish the deployment, then open the project URL above.

## Included files

- `index.html` — project website
- `styles.css` — responsive styling
- `assets/` — project visualizations
- `project/Predictive-Manpower-Planning.ipynb` — Jupyter notebook
- `project/Technical-Support-Dataset.csv` — project dataset
- `.nojekyll` — tells GitHub Pages to serve the files directly

## Methodology note

The site reports the model results saved in the original notebook, while also documenting the target-leakage issue identified during review: `Survey results` is used as a predictor even though the current `Good_Solved` target is directly determined by the presence of survey results. The project therefore treats the perfect model scores as a validation finding rather than evidence of real-world predictive performance.
