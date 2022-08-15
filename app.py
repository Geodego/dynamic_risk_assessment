"""
API set-up for model reporting: scripts that create reports related to the ML model, its performance, and related
diagnostics.

author: Geoffroy de Gournay
date: August 2022
"""

from flask import Flask, request, jsonify
import pandas as pd
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
prod_deployment_path = os.path.join(config['prod_deployment_path'])


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Load the data saved in 'datapath' and calculate related model predictions
    :return:
    string with list of predictions related to data saved in 'datapath'.
    """
    logger.info('running predict')
    data_path = request.get_json()['datapath']

    df = pd.read_csv(data_path)
    y_pred = model_predictions(df)
    return str(y_pred)


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats1():
    """
    Check the f1 score of the deployed model on test data.
    :return:
    f1 score (str)
    """
    logger.info('running stats1')
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    score = score_model(test_data_path, model_path)
    return str(score)


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats2():
    """
    check means, medians, and modes for each column
    :return:
    json dictionary of all calculated summary statistics
    """
    logger.info('running stats2')
    col_stats = dataframe_summary()
    return jsonify(col_stats)


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def stats3():
    """
    check timing and percent NA values, and dependencies
    :return:
    json of:
    dictionary: {
    'missing': missing values statistics,
    'time_check': timing of training and ingestion,
    'outdated': list of packages with version used and latest available version
    }
    """
    logger.info('running stats3')
    missing = missing_data()
    time_check = execution_time()
    outdated = outdated_packages_list()
    diags = {'missing': missing, 'time_check': time_check, 'outdated': outdated}
    return jsonify(diags)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
