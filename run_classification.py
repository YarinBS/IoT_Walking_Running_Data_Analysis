import os
import pickle
import sys
import pandas as pd
from math import sqrt

import warnings
warnings.filterwarnings("ignore")


def run_classification(model_path, sample_path):
    model = pickle.load(open(model_path, 'rb'))
    sample = pd.read_csv(sample_path, skiprows=5)
    sample.name = sample_path[sample_path.rfind('/'):]
    steps_count = int(list(pd.read_csv(sample_path, skiprows=lambda x: x != 3).columns)[1])

    good_columns = ['Time [sec]', 'ACC X', 'ACC Y', 'ACC Z']
    columns_list = list(sample.columns)
    for _ in range(2):
        if columns_list != good_columns:
            if "Unnamed: 0" in columns_list:
                sample.columns = sample.iloc[0]
                sample.drop(index=[0], axis=0, inplace=True)
            else:
                sample.columns = good_columns
            columns_list = list(sample.columns)

    for col in sample.columns:
        sample[col] = sample[col].astype('float64')

    sample = processing(sample, steps=steps_count)
    prediction = model.predict(sample)[0]
    return prediction


def processing(df, steps):
    df['N'] = df.apply(lambda x: sqrt(float(x['ACC X']) ** 2 + float(x['ACC Y']) ** 2 + float(x['ACC Z']) ** 2), axis=1)
    df['steps_count'] = steps

    stats_df = df.describe().T

    accx_mean = stats_df['mean'][1]
    accx_std = stats_df['std'][1]
    accx_min = stats_df['min'][1]
    accx_max = stats_df['max'][1]
    accx_median = stats_df['50%'][1]
    accx_interval = accx_max - accx_min

    accy_mean = stats_df['mean'][2]
    accy_std = stats_df['std'][2]
    accy_min = stats_df['min'][2]
    accy_max = stats_df['max'][2]
    accy_median = stats_df['50%'][2]
    accy_interval = accy_max - accy_min

    accz_mean = stats_df['mean'][3]
    accz_std = stats_df['std'][3]
    accz_min = stats_df['min'][3]
    accz_max = stats_df['max'][3]
    accz_median = stats_df['50%'][3]
    accz_interval = accz_max - accz_min

    N_mean = stats_df['mean'][5]
    N_std = stats_df['std'][5]
    N_min = stats_df['min'][5]
    N_max = stats_df['max'][5]
    N_median = stats_df['50%'][5]
    N_interval = N_max - N_min

    row = [accx_mean, accx_std, accx_min, accx_max, accx_median, accx_interval,
           accy_mean, accy_std, accy_min, accy_max, accy_median, accy_interval,
           accz_mean, accz_std, accz_min, accz_max, accz_median, accz_interval,
           N_mean, N_std, N_min, N_max, N_median, N_interval]

    stats_dataframe = pd.DataFrame([row], columns=['accx_mean', 'accx_std', 'accx_min', 'accx_max', 'accx_median',
                                                 'accx_interval',
                                                 'accy_mean', 'accy_std', 'accy_min', 'accy_max', 'accy_median',
                                                 'accy_interval',
                                                 'accz_mean', 'accz_std', 'accz_min', 'accz_max', 'accz_median',
                                                 'accz_interval',
                                                 'N_mean', 'N_std', 'N_min', 'N_max', 'N_median', 'N_interval'])

    return stats_dataframe


def main(path):
    prediction = run_classification(model_path=r'./models/classification_model.sav', sample_path=path)
    print(f"{'walk' if prediction else 'run'}")
    return prediction


if __name__ == '__main__':
    main(path=sys.argv[1])
    # for file in os.listdir('./data'):
    #     print(file)
    #     try:
    #         main(path=f"./data/{file}")
    #     except ValueError:
    #         continue
