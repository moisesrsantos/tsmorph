# tsMorph

tsMorph is a Python package designed to generate semi-synthetic time series through morphing techniques. It enables the systematic transformation between two given time series, facilitating robust performance evaluation of forecasting models.

This package is based on the paper:  
**Santos, M., de Carvalho, A., & Soares, C. (2024).** *Enhancing Algorithm Performance Understanding through tsMorph: Generating Semi-Synthetic Time Series for Robust Forecasting Evaluation.* [arXiv:2312.01344](https://arxiv.org/abs/2312.01344)

## Features
- **Generation of Semi-Synthetic Time Series**: Creates a set of intermediate time series transitioning from a source series (S) to a target series (T).
- **Performance Understanding**: Evaluates forecasting models' robustness using MASE (Mean Absolute Scaled Error) over synthetic series.
- **Feature Extraction**: Uses `pycatch22` to extract time series features for deeper analysis.
- **Visualization Tools**: Provides plotting functions to explore synthetic time series and their performance.

## Installation

```bash
pip install tsmorph
```

## Usage

### Generate Semi-Synthetic Time Series

```python
import numpy as np
import pandas as pd
from tsmorph import TSmorph

# Define source and target time series
S = np.array([1, 2, 3, 4, 5])
T = np.array([6, 7, 8, 9, 10])

ts_morph = TSmorph(S, T, granularity=5)
synthetic_df = ts_morph.fit()
print(synthetic_df)
```

### Plot Semi-Synthetic Time Series

```python
ts_morph.plot(synthetic_df)
```

### Performance Understanding with Forecasting Models

```python
from some_forecasting_model import TrainedModel

# Assume a trained forecasting model compatible with NeuralForecast
model = TrainedModel()

# Define forecast horizon
horizon = 2

# Analyze performance over synthetic series
ts_morph.analyze_morph_performance(synthetic_df, model, horizon)
```

## Citation
If you use `tsMorph` in your research, please cite:

```bibtex
@article{santos2024tsmorph,
  title={Enhancing Algorithm Performance Understanding through tsMorph: Generating Semi-Synthetic Time Series for Robust Forecasting Evaluation},
  author={Santos, Mois{\'e}s and de Carvalho, Andr{\'e} and Soares, Carlos},
  journal={arXiv preprint arXiv:2312.01344},
  year={2024}
}
```

## Example: Visual comparison — Linear vs DBA 🔧

This repository includes an example that generates a visual comparison between pure linear morphing and morphing with DBA alignment.

- Example script: `examples/compare_morphing.py`
- Generates: `examples/morph_comparison.png`

How to run:

```bash
python examples/compare_morphing.py
```

The script produces a figure with two stacked subplots:
- Top: linear morphing between `S` and `T` (no temporal alignment).
- Bottom: DBA-aligned morphing (series are aligned before interpolation).

It also prints a simple benchmark (execution time for `fit` with and without DBA). Typically, DBA is slower due to the iterative DTW path computations.

Example output (generated image):

![Linear vs DBA comparison](examples/morph_comparison.png)

---

## License
This project is licensed under the GNU General Public License v3.0.

## Funding information

Agenda “Center for Responsible AI”, nr. C645008882-00000055, investment project nr. 62, financed by the Recovery and Resilience Plan (PRR) and by European Union -  NextGeneration EU.

AISym4Med (101095387) supported by Horizon Europe Cluster 1: Health, ConnectedHealth (n.o 46858), supported by Competitiveness and Internationalisation Operational Programme (POCI) and Lisbon Regional Operational Programme (LISBOA 2020), under the PORTUGAL 2020 Partnership Agreement, through the European Regional Development Fund (ERDF)
