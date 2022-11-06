import click
import logging
from pathlib import Path
from loguru import logger

from dvsolution.train import train
from dvsolution.predict import predict
from dvsolution.utils import validate_inputs


@click.command('train', short_help='Trains regressor on data provided')
@click.argument('folder', type=click.Path(exists=True))
@click.option('--task-type', default='CPU', type=str)
def train_regime(folder: str, task_type: str = 'CPU'):
    """  Trains regressor, based on data in provided FOLDER """
    # click.echo(f'Training for {folder}, {task_type}')
    logger.debug(f'Training for {folder}, {task_type}')
    validate_inputs(Path(folder), ['data.h5', 'result.h5'])

    train(
        folder=Path(folder),
        task_type=task_type
    )


@click.command('predict',
               short_help='Predicts returns, based on trained model')
@click.argument('folder', type=click.Path(exists=True))
@click.option('--model-path', default=Path('trained_model'),
              type=click.Path(exists=True))
@click.option('--task-type', default='CPU', type=str)
def predict_regime(folder: str, model_path: click.Path, task_type: str):
    """  Uses regressor to predict data in provided FOLDER """

    logger.debug(f'Predicting for {folder}')
    validate_inputs(Path(folder), ['data.h5'])
    predict(
        folder=Path(folder),
        model_path=Path(model_path),
        task_type=task_type
    )


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    ...


cli.add_command(train_regime)
cli.add_command(predict_regime)
