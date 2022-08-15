"""
Plots generation related to the ML model's performance.

author: Geoffroy de Gournay
date: August 2022
"""

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging

from diagnostics import model_predictions

logger = logging.getLogger(__name__)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)


def get_confusion_matrix():
    """
    Reporting: calculate a confusion matrix using the test data and the deployed model. Write the confusion matrix to
    the directory specified in the output_model_path entry of the config.json file.
    :return:
    """
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    # true labels for test data
    y_true = pd.read_csv(test_data_path)['exited'].values

    # labels predicted by the model
    data = pd.read_csv(test_data_path)
    y_pred = model_predictions(data)

    # plot the confusion matrix
    confusion = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(confusion, annot=True, cmap='Blues')

    ax.set_title('Confusion matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig(os.path.join(config['output_model_path'], 'confusionmatrix.png'))
    logger.info(f"confusion matrix saved in {config['output_model_path']}")


if __name__ == '__main__':
    get_confusion_matrix()
