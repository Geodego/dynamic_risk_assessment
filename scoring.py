import pandas as pd
import pickle
import os
from sklearn import metrics
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')


def score_model():
    """
    Function for model scoring
    :return:
    f1 score (float)
    """
    # load test_data and model
    test_data = pd.read_csv(test_data_path)
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    model = pickle.load(open(model_path, 'rb'))

    # calculate f1 score
    X = test_data.iloc[:, 1:]
    y = X.pop('exited').values
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    # save the result
    output_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    with open(output_path, 'w') as f:
        f.write(str(f1_score) + '\n')
    return f1_score


if __name__ == '__main__':
    score_model()

