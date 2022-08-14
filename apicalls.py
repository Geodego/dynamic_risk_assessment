"""
This script calls each of the API endpoints, combine the outputs, and write the combined outputs to a file called
apireturns.txt.

author: Geoffroy de Gournay
date: August 2022
"""
import pandas as pd
import requests
import os
import json
import logging

logger = logging.getLogger(__name__)

# Specify a URL that resolves to the workspace
URL = "http://127.0.0.1:8000/"

# Get test data:
# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

# path to data used for analysis
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

# Call each API endpoint and store the responses
# predictions
header = {"Content-Type": "application/json"}
body = {'datapath': test_data_path}
response1 = requests.post(
    url=URL + '/prediction',
    headers=header,
    json=body
).text

# scoring
response2 = requests.get(URL + '/scoring').text

# statistics
response3 = requests.get(URL + '/summarystats').json()

# diagnostics
response4 = requests.get(URL + '/diagnostics').json()

# combine all API responses
responses = "-" * 50 + '\n'
responses += " " * 10 + "** Model reporting **\n"
responses += "-" * 50 + '\n\n'
responses += " " * 10 + "Predictions:\n\n"
responses += response1 + '\n'
responses += '-' * 50 + '\n'
responses += " " * 10 + "F1 score:\n\n"
responses += response2 + '\n'
responses += '-' * 50 + '\n'
responses += " " * 10 + "Statistics:\n\n"
df_stats = pd.DataFrame(response3).T
responses += df_stats.to_string() + '\n'
responses += '-' * 50 + '\n'
responses += " " * 10 + "Diagnostics:\n\n"
responses += "missing data per column (%):\n"
responses += json.dumps(response4['missing'], indent=4, sort_keys=True) + '\n\n'
responses += "Ingestion and training execution time:\n"
responses += str(response4['time_check']) + '\n\n'
responses += "outdated packages:\n"
df = pd.DataFrame(response4['outdated']).set_index('module')
responses += df.to_string() + '\n\n'
responses += '-' * 50 + '\n'

report_path = os.path.join(config['output_model_path'], 'apireturns.txt')

with open(report_path, 'w') as f:
    f.write(responses)
