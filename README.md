### Architecture
Boosting on aggregated data, using shuffled cross-validation for model selection and out-of-time (last 100k observations) as performance estimation. Expected quality on out-of-sample:  $R^2 = 0.025$

---

### Installation

In this project Python 3.8.10 has been used.

```bash
git clone https://github.com/danyanyam/test.git;
pip install -e test --quiet
```
---
### Usage

TL;DR: [colab](https://colab.research.google.com/drive/167J6yaqFGRiDLkYrYbwPr23M4wsUOtn2?usp=sharing).

Assuming following folder structure:

```bash
 .
├── ...
└── btcusd-h-30
   ├── data.h5
   └── result.h5

```

Training regression, given folder and using GPU. This command trains and
saves model to `./trained_model`:

```bash
dvsolution train btcusd-h-30 --task-type GPU
```

Using regression, given folder and using GPU. This command uses trained
model at `./trained_model` and produces forecast to `./btcusd-h-30/forecast.h5`

```bash
dvsolution predict btcusd-h-30 --task-type GPU
```