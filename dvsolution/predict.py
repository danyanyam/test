import h5py
from pathlib import Path
from loguru import logger

from dvsolution.train import prepare_features, prepare_target, prepare_model


def predict(
    folder: Path,
    model_path: Path,
    task_type: str = 'CPU',
) -> None:
    """
    Predicts target, given data in folder of the following structure:

    folder
        - data.h5
        - result.h5

    Args:
        folder (Path): data, which consists of orderbooks and target
        model_path (Path): path to pretrained model
        task_type (str, optional): Regime of usage. Either 'CPU' or 'GPU'.
        Defaults to 'CPU'.
    """

    X = prepare_features(folder)
    ts, y = prepare_target(folder)
    model = prepare_model(task_type=task_type)
    model.load_model(model_path)
    preds = model.predict(X)

    f = h5py.File(str(folder / 'forecast.h5'), 'w')
    group = f.create_group('Return')
    group.create_dataset(
        name='Res',
        shape=preds.shape,
        dtype=preds.dtype,
        data=preds
    )
    group.create_dataset(
        name='TS',
        shape=ts.shape,
        dtype=ts.dtype,
        data=ts
    )
    logger.info(f'Predictions have been written to {folder/"forecast.h5"}')
