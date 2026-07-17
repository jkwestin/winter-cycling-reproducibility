# Winter Cycling Reproducibility

Synthetic data, code, and documented workflows reproducing Tables 2 and 3 from the manuscript *Who Cycles in Winter? Seasonal Cycling and Mode Share Change in Northern Sweden*.

## Contents

This repository contains:

- a fully synthetic example dataset with the same variable structure as the confidential travel survey data
- Python code reproducing the weighted mode share changes in Table 2
- Python code estimating the small and full multinomial logit models in Table 3
- person normalized population weights
- person clustered standard errors
- predicted autumn to winter probability changes by gender

The synthetic data contain no actual respondents, locations, or trip records and should not be used for substantive inference.

## Files

```text
data/
    synthetic_data_2022_2025.csv

reproduce_tables_2_3.py

output/
    table2.csv
    table3_coefficients.csv
    table3_predictions.csv
```

## Requirements

The workflow requires Python 3.11 or later and the following packages:

```text
numpy
pandas
scipy
biogeme
```

Create an environment and install the packages with:

```bash
pip install numpy pandas scipy biogeme
```

## Run the workflow

From the repository root, run:

```bash
python reproduce_tables_2_3.py \
    --data data/synthetic_data_2022_2025.csv \
    --output output \
    --threads 8
```

The number of Biogeme threads can be changed with the `--threads` option.

## Outputs

The script creates three CSV files:

- `table2.csv` contains autumn to winter changes in weighted mode shares
- `table3_coefficients.csv` contains model coefficients, person clustered standard errors, test statistics, and p values
- `table3_predictions.csv` contains predicted autumn to winter probability changes for men, women, and the difference between women and men

Because the example data are synthetic, the numerical results may differ from the published tables. The purpose of the repository is to reproduce the analysis workflow and output structure.

## Data availability

The original trip level records cannot be shared because they contain sensitive individual location data and are held under agreement with Umeå Municipality.

The synthetic dataset preserves the variables and structure required to run the code but does not represent real individuals or trips.

## Manuscript

Jonas Westin and Per Åhag. *Who Cycles in Winter? Seasonal Cycling and Mode Share Change in Northern Sweden.* Unpublished manuscript.

The manuscript citation and repository documentation will be updated if the paper is published.

## Citation

When using these materials, please cite the archived release of this repository. The unpublished manuscript may also be cited where appropriate. A complete publication reference will be added if the paper is published.

## License

The Python code is licensed under the MIT License.

The synthetic dataset and documentation are licensed under the Creative Commons Attribution 4.0 International License.

See `LICENSE` and `LICENSE-DATA` for details.
