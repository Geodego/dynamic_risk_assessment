"""
Data ingestion process.

author: Geoffroy de Gournay
date: August 2022
"""
import pandas as pd
import os
import json
import logging

logger = logging.getLogger(__name__)

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    """
    Function for data ingestion. Read all files in input_folder_path, concatenate them as a single dataframe, remove
    duplicates  and save the data in output_folder_path/finaldata.csv
    """
    logger.info('starting data ingestion process')
    # check for datasets
    filenames = next(os.walk(input_folder_path), (None, None, []))[2]  # [] if no file

    # compile the datasets together
    data_list = []
    for file in filenames:
        data_list.append(pd.read_csv(os.path.join(input_folder_path, file)))
    data = pd.concat(data_list)

    # remove duplicates
    data = data.drop_duplicates(ignore_index=True)

    # Write to an output file
    data_path = os.path.join(output_folder_path, 'finaldata.csv')
    try:
        data.to_csv(data_path, index=False)
    except FileNotFoundError:
        os.mkdir(output_folder_path)
        data.to_csv(data_path, index=False)

    # saving a record of the ingestion
    record_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(record_path, 'w') as f:
        for file in filenames:
            f.write(file + '\n')
    logger.info(f"record of ingestion saved in {record_path}")

    return data


if __name__ == '__main__':
    merge_multiple_dataframe()
