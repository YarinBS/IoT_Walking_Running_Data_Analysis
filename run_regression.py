import pickle
import sys
import pandas as pd


def run_regression(model_path, sample_path):
    model = pickle.load(open(model_path, 'rb'))
    sample = pd.read_csv(sample_path)
    prediction = model.predict(sample)
    return prediction


def main(path):
    prediction = run_regression(model_path=r'./models/regression_model.sav', sample_path=path)
    print(prediction)


if __name__ == '__main__':
    main(path=sys.argv[1])
