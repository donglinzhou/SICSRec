import torch
import optuna
import argparse
from logging import getLogger
from recbole.config import Config
from TICRec import Content_SeqEncoder
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from data.tiger_dataset import UniSRecDataset


def load_data(config):
    # Dataset loading and splitting
    dataset = UniSRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return dataset, train_data, valid_data, test_data

def run_baseline(config, train_data, valid_data, test_data):
    # Logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    logger.info(train_data.dataset)

    # Model initialization
    model = Content_SeqEncoder(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # Trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # Model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return best_valid_score, test_result

def objective(trial, args, train_data, valid_data, test_data, config):
    alpha = trial.suggest_categorical('alpha', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    config['alpha'] = alpha
    init_seed(config['seed'], config['reproducibility'])

    best_valid_score, _ = run_baseline(config, train_data, valid_data, test_data)
    return best_valid_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Content_SeqEncoder', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Bili_Cartoon', help='name of datasets')
    parser.add_argument('--n_trials', type=int, default=10, help='number of trials for optimization')
    args, _ = parser.parse_known_args()

    # Configuration
    # configurations initialization
    config_file_list = [f'props/{args.model}.yaml', 'props/finetune.yaml']
    config = Config(model=Content_SeqEncoder, dataset=args.dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # Load dataset once
    dataset, train_data, valid_data, test_data = load_data(config)

    # Create a study with a grid sampler
    sampler = optuna.samplers.GridSampler(
        search_space={
         'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, args, train_data, valid_data, test_data, config),
                   n_trials=args.n_trials)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')