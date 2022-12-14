"""
Model and Data Diagnostics: diagnostic tests related to the model as well as the data.

author: Geoffroy de Gournay
date: August 2022
"""

import subprocess

import pandas as pd
import timeit
import os
import json
import pickle
import logging

logger = logging.getLogger(__name__)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')


def model_predictions(data):
    """
    Get model predictions: read the deployed model and a test dataset, calculate predictions
    :param data: data we use for prediction represented as a panda Dataframe
    :return:
    list containing all predictions
    """
    logger.info('calculate model predictions')
    production_path = os.path.join(config['prod_deployment_path'])
    model_path = os.path.join(production_path, 'trainedmodel.pkl')
    model = pickle.load(open(model_path, 'rb'))
    X = data.iloc[:, 1:-1]
    predictions = list(model.predict(X))
    return predictions


def dataframe_summary():
    """
    Get summary statistics
    :return:
    dictionary of statistics (mean, median, std deviation) related to each numerical column
    """
    logger.info('calculate statistics on the data')
    data = pd.read_csv(dataset_csv_path)
    X = data.iloc[:, 1:-1]
    means = X.mean()
    medians = X.median()
    std_var = X.std()

    col_stats = {}
    for col in X.columns:
        col_stats[col] = {'mean': means[col], 'median': medians[col], 'std_dev': std_var[col]}

    return col_stats


def missing_data():
    """
    Check for missing data by calculating what percent of each column consists of NA values.
    :return:
    Dictionary with keys corresponding to the columns of the dataset.
    Each element of the dictionary gives the percent of NA values in a particular column of the data.
    """
    logger.info('check for missing data')
    data = pd.read_csv(dataset_csv_path)
    missing = data.isna().sum()
    n_data = data.shape[0]
    missing = missing / n_data
    return missing.to_dict()


def execution_time():
    """
    Get timings: calculate timing of training.py and ingestion.py.
    :return:
    list of 2 timing values in seconds
    """
    logger.info('calculate timing for ingestion and training')
    # timing ingestion
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing = timeit.default_timer() - starttime

    # timing training
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - starttime

    return [ingestion_timing, training_timing]


def outdated_packages_list():
    """
    Check dependencies: checks the current and latest versions of all the modules that the scripts use
    (the current version is recorded in requirements.txt).

    :return:
    Output a list of dictionaries, one for each package used: the first key will show the name of a Python
    module that is used; the second key will show the currently installed version of that Python module, and
    the third key will show the most recent available version of that Python module:
    [{'module': 'click', 'current': '7.1.2', 'latest': '8.1.3'}, ...]
    """
    logger.info('Check dependencies versions')
    # current version of dependencies
    with open('requirements.txt', 'r') as req_file:
        requirements = req_file.read().split('\n')
    requirements = [r.split('==') for r in requirements if r]
    df = pd.DataFrame(requirements, columns=['module', 'current'])

    # Get outdated dependencies using PIP
    outdated_dep = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf8')
    outdated_dep = outdated_dep.split('\n')[2:]  # the first 2 items are not packages
    outdated_dep = [x.split(' ') for x in outdated_dep if x]
    outdated_dep = [[y for y in x if y] for x in outdated_dep]  # list of [package, current version, latest version]
    outdated_dic = {x[0]: x[2] for x in outdated_dep}  # {package: latest version}
    df['latest'] = df['module'].map(outdated_dic)

    # if we're already using the latest version of a module, we fill latest with this version:
    df['latest'].fillna(df['current'], inplace=True)
    return df.to_dict('records')


if __name__ == '__main__':
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    data = pd.read_csv(test_data_path)
    y_pred = model_predictions(data)
    stats = dataframe_summary()
    missing = missing_data()
    time_check = execution_time()
    outdated = outdated_packages_list()
    pass
