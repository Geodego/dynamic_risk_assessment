"""
Model deployment.

author: Geoffroy de Gournay
date: August 2022
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def deploy_model():
    """
    function for deployment, copy the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file
    into the deployment directory
    """
    logger.info(
        f"Model deployment. Model, it's latest score, and the list of files used for training are saved in "
        f"{prod_deployment_path}")
    # source paths
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    ingested_path = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    sources = [model_path, score_path, ingested_path]

    # target paths
    # make sure the target directory exists
    if not os.path.isdir(prod_deployment_path):
        os.mkdir(prod_deployment_path)
    target_model = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    target_score = os.path.join(prod_deployment_path, 'latestscore.txt')
    target_ingested = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    targets = [target_model, target_score, target_ingested]

    for source, target in zip(sources, targets):
        os.system(f'cp {source} {target}')


if __name__ == '__main__':
    deploy_model()
