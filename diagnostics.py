import subprocess

import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
production_path = os.path.join(config['prod_deployment_path'])


def model_predictions():
    """
    Get model predictions: read the deployed model and a test dataset, calculate predictions
    :return:
    list containing all predictions
    """
    model_path = os.path.join(production_path, 'trainedmodel.pkl')
    model = pickle.load(open(model_path, 'rb'))
    test_data = pd.read_csv(test_data_path)
    X = test_data.iloc[:, 1:-1]
    predictions = list(model.predict(X))
    return predictions


def dataframe_summary():
    """
    Get summary statistics
    :return:
    list containing pandas Series with columns statistics: mean, median, std deviation
    """
    data = pd.read_csv(dataset_csv_path)
    X = data.iloc[:, 1:-1]

    return [X.mean(), X.median(), X.std()]


def missing_data():
    """
    Check for missing data by calculating what percent of each column consists of NA values.
    :return:
    list with the same number of elements as the number of columns in the dataset.
    Each element of the list will be the percent of NA values in a particular column of the data.
    """
    data = pd.read_csv(dataset_csv_path)
    missing = data.isna().sum()
    n_data = data.shape[0]
    missing = missing / n_data
    return list(missing)


def execution_time():
    """
    Get timings: calculate timing of training.py and ingestion.py.
    :return:
    list of 2 timing values in seconds
    """
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
    Output a pandas Dataframe with three columns: the first column will show the name of a Python
    module that is used; the second column will show the currently installed version of that Python module, and
    the third column will show the most recent available version of that Python module.
    """
    # current version of dependencies
    with open('requirements.txt', 'r') as req_file:
        requirements = req_file.read().split('\n')
    requirements = [r.split('==') for r in requirements if r]
    df = pd.DataFrame(requirements, columns=['module', 'current'])

    # Get outadated dependencies using PIP
    outdated = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf8')
    outdated = outdated.split('\n')[2:]  # the first 2 items are not packages
    outdated = [x.split(' ') for x in outdated if x]
    outdated = [[y for y in x if y] for x in outdated]  # list of [package, current version, latest version]
    outdated_dic = {x[0]: x[2] for x in outdated}  # {package: latest version}
    df['latest'] = df['module'].map(outdated_dic)

    # if we're already using the latest version of a module, we fill latest with this version:
    df['latest'].fillna(df['current'], inplace=True)
    return df


if __name__ == '__main__':
    y_pred = model_predictions()
    stats = dataframe_summary()
    missing = missing_data()
    time_check = execution_time()
    outdated = outdated_packages_list()
    pass
