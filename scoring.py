"""
Model scoring.

author: Geoffroy de Gournay
date: August 2022
"""

import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import logging

logger = logging.getLogger(__name__)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')


def score_model(data_path, model_path, save_result=True):
    """
    Function for model scoring
    :param data_path: path to the data used for scoring
    :param model_path: path to the model used for scoring
    :param save_result: save result in a file if true
    :return:
    f1 score (float)
    """
    # load test_data and model
    test_data = pd.read_csv(data_path)
    model = pickle.load(open(model_path, 'rb'))

    # calculate f1 score
    X = test_data.iloc[:, 1:]
    y = X.pop('exited').values
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    if not save_result:
        return f1_score
    # save the result
    output_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    with open(output_path, 'w') as f:
        f.write(str(f1_score) + '\n')
    logger.info(f'f1 score saved in {output_path}')
    return f1_score


if __name__ == '__main__':
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    model_path1 = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
    f1 = score_model(test_data_path, model_path1, save_result=False)
    print(f1)

