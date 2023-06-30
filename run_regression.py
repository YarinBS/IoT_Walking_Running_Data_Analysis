import pickle
import sys
import pandas as pd
from math import sqrt


def run_regression(model_path, sample_path):
    model = pickle.load(open(model_path, 'rb'))
    sample = pd.read_csv(sample_path, skiprows=5)
    sample.name = sample_path[sample_path.rfind('/'):]
    sample = processing(sample)
    prediction = model.predict(sample)
    return prediction


def processing(df):
    df['N'] = df.apply(lambda x: sqrt(x['ACC X'] ** 2 + x['ACC Y'] ** 2 + x['ACC Z'] ** 2), axis=1)

    stats_df = df.describe().T

    number_of_timestamps = stats_df['count'][0]
    recording_duration = stats_df['max'][0] - stats_df['min'][0]

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

    activity_type = 1 if 'walk' in df.name else 0

    row = [number_of_timestamps, recording_duration,
           accx_mean, accx_std, accx_min, accx_max, accx_median, accx_interval,
           accy_mean, accy_std, accy_min, accy_max, accy_median, accy_interval,
           accz_mean, accz_std, accz_min, accz_max, accz_median, accz_interval,
           N_mean, N_std, N_min, N_max, N_median, N_interval, activity_type]

    stats_dataframe = pd.DataFrame(row,
                                   columns=['number_of_timestamps', 'recording_duration',
                                            'accx_mean', 'accx_std', 'accx_min', 'accx_max', 'accx_median',
                                            'accx_interval',
                                            'accy_mean', 'accy_std', 'accy_min', 'accy_max', 'accy_median',
                                            'accy_interval',
                                            'accz_mean', 'accz_std', 'accz_min', 'accz_max', 'accz_median',
                                            'accz_interval',
                                            'N_mean', 'N_std', 'N_min', 'N_max', 'N_median', 'N_interval',
                                            'activity_type'])

    return stats_dataframe


def main(path):
    prediction = run_regression(model_path=r'./models/regression_model.sav', sample_path=path)
    print(prediction)


if __name__ == '__main__':
    main(path=sys.argv[1])
