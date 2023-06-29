import pickle
import sys
import pandas as pd


def run_classification(model_path, sample_path):
    model = pickle.load(open(model_path, 'rb'))
    sample = pd.read_csv(sample_path)
    prediction = model.predict(sample)
    return prediction


def main(path):
    prediction = run_classification(model_path=r'./models/classification_model.sav', sample_path=path)
    print(f"{'walk' if prediction else 'run'}")


if __name__ == '__main__':
    main(path=sys.argv[1])
