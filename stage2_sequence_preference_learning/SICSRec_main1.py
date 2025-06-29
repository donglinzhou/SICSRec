import torch
import optuna
import argparse
from SICSRec import SICSRec
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from data.tiger_dataset import UniSRecDataset


def load_data(config):
    # Dataset loading and splitting
    dataset = UniSRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return dataset, train_data, valid_data, test_data

def run_baseline(config, train_data, valid_data, test_data, pretrain_id='', pretrain_content=''):
    # Logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    logger.info(train_data.dataset)

    # Model initialization
    model = SICSRec(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # Load id encoder
    if pretrain_id != '':
        checkpoint = torch.load(pretrain_id, map_location=config['device'])
        print(f'Loading id-based local model from {pretrain_id}')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)

        print(f'Fix encoder parameters.')
        for param in model.item_embedding.parameters():
            param.requires_grad = True
        for param in model.trm_encoder.parameters():
            param.requires_grad = False

        for layer in model.trm_encoder.layer:
            for param in layer.att_lora_layer.parameters():
                param.requires_grad = True
            for param in layer.att_lora_layer.parameters():
                param.requires_grad = True

    # if pretrain_content != '':
    #     checkpoint = torch.load(pretrain_content, map_location=config['device'])
    #     print(f'Loading id-based local model from {pretrain_content}')
    #     state_dict = checkpoint['state_dict']
    #     model.load_state_dict(state_dict, strict=False)
    #     print(f'Fix content encoder parameters.')
    #
    #     for param in model.content_trm_decoder.parameters():
    #         param.requires_grad = False

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
    # alpha = trial.suggest_categorical('alpha', [0.25, 0.5, 0.75, 1.0])
    # beta = trial.suggest_categorical('beta', [0.25, 0.5, 0.75, 1.0])
    # lora_rank = trial.suggest_categorical('lora_rank', [4, 8, 16])
    # config['alpha'] = alpha
    alpha = trial.suggest_categorical('alpha', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    config['alpha'] = alpha
    # # config['lora_rank'] = lora_rank
    # lora_rank = trial.suggest_categorical('lora_rank', [4, 8, 12, 16])
    # config['lora_rank'] = lora_rank

    init_seed(config['seed'], config['reproducibility'])
    best_valid_score, test_result = run_baseline(config, train_data, valid_data, test_data,args.pretrain_id, args.pretrain_content)

    # 保存当前试验的参数和结果
    trial_result = {
        'params': trial.params,
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
    all_trial_results.append(trial_result)

    return best_valid_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SICSRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Bili_Cartoon', help='name of datasets')
    parser.add_argument('-pretrain_id', type=str, default='./saved/ID_SeqEncoder-Cartoon.pth')
    parser.add_argument('-pretrain_content', type=str, default='./saved/Content_SeqEncoder-Cartoon.pth')
    parser.add_argument('--n_trials', type=int, default=10, help='number of trials for optimization')
    args, _ = parser.parse_known_args()

    # Configuration
    config_file_list = [f'props/SICSRec.yaml', 'props/finetune.yaml']
    config = Config(model=SICSRec, dataset=args.dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # Load dataset once
    dataset, train_data, valid_data, test_data = load_data(config)

    all_trial_results = []
    # Create a study with a grid sampler
    sampler = optuna.samplers.GridSampler(
        search_space={
            # 'alpha': [0.25, 0.5, 0.75, 1.0],
            # 'beta':[0.25, 0.5, 0.75, 1.0],
            'alpha':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # "lora_rank": [4, 8, 12, 16]
        }
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, args, train_data, valid_data, test_data, config),
                   n_trials=args.n_trials)

    # 输出所有试验结果
    print('\nAll trial results:')
    for i, result in enumerate(all_trial_results, 1):
        print(f'Trial {i}:')
        print(f'  Params: {result["params"]}')
        print(f'  Best valid score: {result["best_valid_score"]}')
        print(f'  Best test score: {result["test_result"]}')