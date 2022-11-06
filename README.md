### Installation

```bash
git clone https://github.com/danyanyam/test.git;
pip install -e test --quiet
```

### Usage

Example on [colab](https://colab.research.google.com/drive/167J6yaqFGRiDLkYrYbwPr23M4wsUOtn2?usp=sharing).

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
model at `./trained_model`

```bash
dvsolution predict btcusd-h-30 --task-type GPU
```
