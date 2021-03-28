import os
import warnings

import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import poisson
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


class View(object):
    def __init__(self, start_date: str, end_date: str, mold_type: str):
        # Data options
        self.mold_type_col = 'mold_type_a'
        self.start_col = 'work_sdt'
        self.end_col = 'work_fdt'
        self.data_name = 'hist.csv'
        self.target_col = ['ori_mold_crft_wth', 'ori_mold_ass_wth', 'ori_mold_mody_cnt', 'ori_mold_mody_wth',
                           'ori_tinj_wth', 'ori_tment_wth']
        self.start_date = start_date
        self.end_date = end_date
        self.mold_type = mold_type

        # Path options
        self.path_data = os.path.join('..', 'data', self.data_name)
        self.path_save = os.path.join('..', 'img', 'hist')
        self.path_save_pois = os.path.join('..', 'img', 'poisson')

        # Histogram options
        self.drop_cols = ['order_no']
        self.bins = 100

        # Poission options
        self.x_start = 0
        self.x_end = 100
        self.interval = 1
        self.lamb = 1.0

    def load_dataset(self):
        df = pd.read_csv(self.path_data)
        df.columns = [col.lower() for col in df.columns]

        # Filter Mold Type
        df = df[df[self.mold_type_col] == self.mold_type]

        # Filter datetime
        start_date = dt.datetime.strptime(self.start_date, '%Y%m%d')
        end_date = dt.datetime.strptime(self.start_date, '%Y%m%d')

        # Need to check
        # df = df[df[self.start_col] >= start_date]
        # df = df[df[self.end_col] <= end_date]

        # Filter date type
        data_type_col = [col for col, typ in zip(df.columns, df.dtypes) if typ in ['int64', 'float64']]
        # Filter dataset
        data_type_col = [col for col in data_type_col if col in self.target_col]

        return df, data_type_col

    def draw_hist(self, df: pd.DataFrame, columns: list, types: list):

        for col in columns:
            data = df[[col, self.mold_type_col]]
            bins = np.linspace(df[col].min(), df[col].max(), self.bins)
            bin_avg = bins[0: -1] + bins[1] - bins[0]

            data_split = {}
            for typ in types:
                temp = data[data[self.mold_type_col] == typ]
                data_split[typ] = np.histogram(temp[col], bins)

            plt.figure()
            for typ in types:
                plt.plot(bin_avg, data_split[typ][0], alpha=0.7, label=typ,
                         marker='o', markersize=2, linewidth=2.0)
            plt.xlabel('Bins')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.path_save, col + '.png'))
            plt.close()

    def draw_poisson(self, df: pd.DataFrame, columns: list):
        for col in columns:
            data = df[col]
            # xvals = np.arange(df[col].min(), df[col].max(), self.interval)
            mu = data.mean()
            x = np.arange(self.x_start, self.x_end, self.interval)
            y = poisson.pmf(x, mu=mu, loc=40)

            plt.figure()
            plt.plot(x, y, alpha=0.7, label=self.mold_type, linewidth=2.0)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(self.path_save_pois, col + '.png'))
            plt.close()


def main():
    view = View(start_date='',
                end_date='',
                mold_type='D')

    df, data_type_col = view.load_dataset()
    print("Dataset is loaded")

    # view.draw_hist(df=df, columns=data_type_col, types=types)
    view.draw_poisson(df=df, columns=data_type_col)
    print("Drawing Histogram is finished")

main()