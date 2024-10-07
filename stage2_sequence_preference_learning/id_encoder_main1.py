import argparse
from logging import getLogger
from TICRec import ID_SeqEncoder
from recbole.config import Config
from recbole.data import data_preparation
from data.tiger_dataset import IDDataset
from recbole.utils import init_seed, init_logger, get_trainer

def run_baseline(config):
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = IDDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = ID_SeqEncoder(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(f'Best valid result: {best_valid_result}')
    logger.info(f'Test result: {test_result}')

    return best_valid_score, test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ID_SeqEncoder', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Bili_Cartoon', help='name of datasets')
    args = parser.parse_args()

    # configurations initialization
    config_file_list = [f'props/{args.model}.yaml', 'props/finetune.yaml']

    # configurations initialization
    config = Config(model=ID_SeqEncoder, dataset=args.dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    best_valid_score, _ = run_baseline(config)


