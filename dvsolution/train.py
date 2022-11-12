import h5py
import numpy as np
import catboost as cb
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple
from sklearn.metrics import r2_score


def prepare_features(folder: Path) -> np.ndarray:
    """Aggregates raw data. Extracts following features:

        - Log of Last trade price
        - Last trade amount
        - Log of total askV
        - Log of total bidV
        - Log of mid-price
        - 10 levels of Ask -- Bid differences
        - 7 levels of AskV -- BidV differences
        - 10 levels of Log AskV -- BidV relation

    Args:
        folder (Path): path to raw data

    Returns:
        np.ndarray: matrix of features
    """
    logger.debug('prepairing features')

    data = h5py.File(str(folder / 'data.h5'), 'r')

    ob_ts = np.array(data['OB/TS'])
    tr_ts = np.array(data['Trades/TS'])

    trade_idx = (np.searchsorted(tr_ts, ob_ts, side='right') - 1).astype(int)
    last_amount = np.array(data['Trades/Amount'])[trade_idx].reshape(-1, 1)
    last_price = np.array(data['Trades/Price'])[trade_idx].reshape(-1, 1)

    return np.hstack(
        (
            np.log(np.array(data['OB/AskV']).sum(axis=1)).reshape(-1, 1),
            np.log(np.array(data['OB/BidV']).sum(axis=1)).reshape(-1, 1),
            np.log(
                (
                    np.array(data['OB/Ask']).min(axis=1) +
                    np.array(data['OB/Bid']).max(axis=1)
                ) / 2
            ).reshape(-1, 1),
            last_amount,
            np.log(last_price),
            np.log(data['OB/Ask'][:, :10] - data['OB/Bid'][:, :10]),
            data['OB/AskV'][:, :7] - data['OB/BidV'][:, :7],
            np.log(data['OB/AskV'][:, :8] / data['OB/BidV'][:, :8]),

        )
    )


def train_test_oot(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[cb.Pool, cb.Pool, cb.Pool, np.ndarray]:
    """Splits data for train 50%, test 25% and out-of-time 25%

    Args:
        X (np.ndarray): All features
        y (np.ndarray): target

    Returns:
        Tuple[cb.Pool, cb.Pool, cb.Pool, np.ndarray]: data
        for training, testing and evaluation
    """

    np.random.seed(69)

    # исключаем out of time часть выборки из выборкм для обучения
    oot_set = np.array(list(range(len(X) - 2_467_910, len(X))))
    X_oot = X[oot_set]
    y_oot = y[oot_set]

    validation_set = np.array(list(range(len(X) - 2*2_467_910,
                                         len(X) - 2_467_910)))
    X_val = X[validation_set]
    y_val = y[validation_set]

    validation_set_ = np.hstack([validation_set, oot_set])
    train_inds = np.array(list(set(range(len(X))) - set(validation_set_)))
    X = X[train_inds]
    y = y[train_inds]

    cols = ['log_asks_v', 'log_bids_v', 'mid_price', 'last_amount',
            'last_price'] +\
        [f'{i}_price' for i in range(10)] +\
        [f'{i}_vol_diff' for i in range(7)] +\
        [f'{i}_vol_rel' for i in range(8)]

    train = cb.Pool(X, y).set_feature_names(cols)
    test = cb.Pool(X_val, y_val).set_feature_names(cols)
    oot = cb.Pool(X_oot, y_oot).set_feature_names(cols)

    return train, test, oot, y_oot


def prepare_target(folder: Path) -> Tuple[h5py._hl.dataset.Dataset]:
    """Reads hdf5 target

    Args:
        folder (Path): path to target file

    Returns:
        Tuple[h5py._hl.dataset.Dataset]: dataset
    """
    logger.debug('prepairing target')
    result = h5py.File(str(folder / 'result.h5'), 'r')
    return result['Return/TS'], result['Return/Res']


def evaluate(
    model: cb.CatBoostRegressor,
    oot:   cb.Pool,
    y_oot: np.ndarray
) -> Dict[str, float]:
    """Evaluates the model, provided data

    Args:
        model (cb.CatBoostRegressor): model to evaluate
        oot (cb.Pool): oot data pool
        y_oot (np.ndarray): target on out of time data

    Returns:
        Dict[str, float]: R2 of model
    """

    logger.debug('evaluating model')
    return {
        'oot': r2_score(y_oot, model.predict(oot)),
        'test': model.get_best_score()['validation']['R2'],
        'train': model.get_best_score()['learn']['R2']
    }


def prepare_model(task_type: str = 'CPU') -> cb.CatBoostRegressor:
    """Initializes model with specified task_type

    Args:
        task_type (str, optional): Which device to use. Either 'CPU' or 'GPU'.
            Defaults to 'CPU'.

    Returns:
        cb.CatBoostRegressor: initted model
    """
    logger.debug('prepairing model')
    return cb.CatBoostRegressor(
        depth=6,
        verbose=5,
        random_seed=69,
        eval_metric='R2',
        iterations=300,
        learning_rate=0.007,
        task_type=task_type,
        bagging_temperature=0.07
    )


def train(folder: Path, task_type: str = 'CPU') -> None:
    """Trains model, given data in folder of the following structure:

    folder
        - data.h5
        - result.h5

    Args:
        folder (Path): path to data
        task_type (str, optional): Which device to use. Either 'CPU' or 'GPU'
            Defaults to 'CPU'.
    """
    logger.info(f'training for folder {folder} has started!')
    X = prepare_features(folder)
    _, y = prepare_target(folder)

    train, test, oot, y_oot = train_test_oot(X, y)
    model = prepare_model(task_type)
    model.fit(train, eval_set=test)
    print(evaluate(model, oot, y_oot))

    model.save_model("trained_model")
    logger.info(f'Training ended! Model has been saved to ./trained_model')
