import logging
from polyaxon_client.tracking import Experiment, get_log_level, get_outputs_path
import os
from ..datapipeline.datapipeline import Datapipeline
from src.modelling.model import twitter_model
import argparse
import nltk

logger = logging.getLogger(__name__)


def run_experiment(data_path,glove_path):
    # try:
    log_level = get_log_level()
    if not log_level:
        log_level = logging.INFO

    logger.info("Starting experiment")

    experiment = Experiment()
    logging.basicConfig(level=log_level)

    logging.info("Loading data")
    dpl = Datapipeline(data_path=data_path)
    dpl.transform()
    train, val = dpl.split_data()
    logging.info("Data loaded")
    model = twitter_model(glove_path=glove_path)
    model.build_model(train.values)
    model.get_train_data(train.values)
    output_model = model.train()

    filepath = os.path.join(get_outputs_path(), "trump_bot.h5")

    # metrics = model.train(params)
    #
    # experiment.log_metrics(**metrics)
    # save model
    output_model.save(filepath)

    logger.info("Experiment completed")
    # except Exception as e:
    #     logger.error(f"Experiment failed: {str(e)}")


if __name__ == '__main__':
    try:
        nltk.download('wordnet')
    except:
        os.environ['NLTK_DATA'] = '/polyaxon-data/workspace/jin_howe_teo/assignment8/nltk_data/'

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', dest='data_path')
    parser.add_argument('-g', dest='glove_path', type=str, default='./data/glove.6B.100d.txt')

    args = parser.parse_args()
    #in_params = {}  # Add your own params
    run_experiment(data_path=args.data_path, glove_path=args.glove_path)
