import h5py
import numpy as np
import catboost as cb
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def prepare_features(folder: Path) -> np.ndarray:
    logger.debug('prepairing features')

    data = h5py.File(str(folder / 'data.h5'), 'r')
    mid_price = (
        np.array(data['OB/Ask']).min(axis=1) +
        np.array(data['OB/Bid']).max(axis=1)
    ) / 2

    return np.hstack(
        (
            np.array(data['OB/Ask'][:, :5]) / mid_price,
            np.array(data['OB/Bid'][:, :5]) / mid_price,
            np.array(data['OB/BidV'][:, :5]) / mid_price,
            np.array(data['OB/AskV'][:, :5]) / mid_price,
            mid_price,
        )
    )


def prepare_target(folder: Path) -> Tuple[h5py._hl.dataset.Dataset]:
    logger.debug('prepairing target')
    result = h5py.File(str(folder / 'result.h5'), 'r')
    return result['Return/TS'], result['Return/Res']


def eval(
    model: cb.CatBoostRegressor,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_oot: np.ndarray,
    y_oot: np.ndarray
) -> Dict[str, float]:
    logger.debug('evaluating model')
    return {
        'val': r2_score(y_val, model.predict(X_val)),
        'oot': r2_score(y_oot, model.predict(X_oot)),
        'test': model.get_best_score()['validation']['R2'],
        'train': model.get_best_score()['learn']['R2']
    }


def prepare_model(task_type: str = 'CPU') -> cb.CatBoostRegressor:
    logger.debug('preparing model')
    return cb.CatBoostRegressor(
        depth=7,
        verbose=50,
        random_seed=69,
        eval_metric='R2',
        learning_rate=0.15,
        task_type=task_type,
        bagging_temperature=0.15
    )


def train(folder: Path, task_type: str = 'CPU') -> None:
    logger.info(f'training for folder {folder} has started!')
    X = prepare_features(folder)
    _, y = prepare_target(folder)

    X_oot, X = X[-10_000:], X[:-10_000]
    y_oot, y = y[-10_000:], y[:-10_000]

    model = prepare_model(task_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69)
    model.fit(X_train, y_train, eval_set=cb.Pool(X_test, y_test))

    print(eval(model, X_test, y_test, X_oot, y_oot))
    model.save_model("trained_model")
    logger.inf(f'Training ended! Model has been saved to ./trained_model')
