import os
import copy
import warnings
from typing import Dict
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')    # Time Series Theme


class DataAnalysis(object):
    """
    Data Analysis step
    ---------------------------------
    1. Load dataset
        - work : work information
        - mold : mold information
    2. Data Preprocessing
    3. Draw Graph
        - Line Plot :
        - Line Plot :
        - BOx Plot :
    """
    def __init__(self):
        # Load Dataset
        self.work_data_ver = '0.4'   # 2021.2.5 16:13
        self.date_cols = ['work_sdt', 'work_fdt']    # Datetime format columns

        # Data Preprocessing
        self.rename_col_mold = {'금형번호': 'mold_id', '기종 도번': 'draw_id', '패키지 도번': 'pkg_id',
                                '제품 도번': 'prod_id', '제품 상세 도번': 'prod_id_dtl', '패키지': 'pkg',
                                '원재료': 'raw_matl', '타입': 'mold_type', '기종': 'mac_type', 'erp 등록일': 'erp_reg_day',
                                '최종 수정일': 'fin_mod_day', '제품 도면': 'prod_draw_day', '금형 도면': 'mold_draw_day',
                                '제작일(추정)': 'make_day', '현물 현황': 'cash_status'}
        self.subset_col_work = ['order_no', 'mold_id', 'work_sdt', 'work_fdt']    # Necessary columns on work dataset
        self.subset_col_mold = ['mold_id', 'draw_id', 'mold_type']      # Necessary columns on mold dataset
        self.agg_list = ['sum', 'avg']    # Applying Aggregation List

        # Draw Graph
        self.mold_group = [['6', '7', '80', '81', '82', '83', '85', '86', '87', '88', '8A', '8B',
                           '8C0', '8C1', '8C2', '8C3', '8C4', '8C5', '8C6', '8C7', '8D']]

    def load_dataset(self):
        print("Process: Load Dataset")
        # Load dataset
        # Work data
        path_work = os.path.join('..', 'data', f'data_v{self.work_data_ver}.csv')
        work = self._load_data(path=path_work, header=0)

        # Mold data
        path_mold = os.path.join('..', 'data', 'mold_info.csv')
        mold = self._load_data(path=path_mold, header=0, delimiter='\t')
        mold = mold.rename(columns=self.rename_col_mold)

        return work, mold

    def data_prep(self, work: pd.DataFrame, mold: pd.DataFrame):
        print("Process: Data Preprocessing")

        # ----------------------
        # Preprocessing : work
        # ----------------------
        # Convert date data to datetime
        work = self._conv_to_datetime(df=work)

        # Rename columns
        work = work.rename(columns={'mold_no': 'mold_id'})

        # Filter necessary data
        work = work[self.subset_col_work]

        # Calculate Making times
        work = self._calc_make_time(df=work, name='make_time')

        # Group and aggregate
        agg = self._aggregate(df=work, by='date', agg_col='make_time')     # date aggregation
        # agg_mold = self._aggregate(work=df_day, by='mold_id', agg_col='make_time')    # mold

        # Make times of each mold
        mold_time = self._group_list(df=work, key='mold_id', val='make_time')
        # mold_time = df_day.groupby(by='mold_id')['make_time'].apply(list)
        mold_time_split = self._split_by_group(df=mold_time, group=self.mold_group, split_col='mold_id')

        # ----------------------
        # Preprocessing : mold
        # ----------------------
        # Filter necessary data
        mold = mold[self.subset_col_mold]

        # Remove exception
        mold['mold_id'] = mold['mold_id'].str[1:]   # Remove '#' from mold_id

        # Re-group
        mold['draw_id'] = mold['draw_id'].str.split('-').str[-2]
        mold['draw_id'] = mold['draw_id'].str[0]

        # Re-group mold types
        grp_name = 'mold_group'
        mold = self._re_grp_mold_type(df=mold, name=grp_name)

        # Remove non-group
        mold = mold[~mold[grp_name].isin([0, np.nan])]

        # Merge dataset
        mold_group = pd.merge(work, mold, how='left', on='mold_id', left_index=True, right_index=False)

        # Filter & slice group mapping data
        mold_group = mold_group.dropna(axis=0, how='any', subset=[grp_name])
        subset_col = copy.deepcopy(self.subset_col_work)
        subset_col.append(grp_name)
        mold_group = mold_group[subset_col]

        # Calculate Making times
        mold_group = self._calc_make_time(df=mold_group, name='make_time')

        # Make times of each mold
        mold_group = self._group_list(df=mold_group, key=grp_name, val='make_time')
        mold_group_split = self._split_by_group(df=mold_group, group=[list(mold_group[grp_name])], split_col=grp_name)

        return agg, mold_time_split, mold_group_split

    def draw_graph_line(self, line_plot: dict) -> None:
        self._chk_and_make_dir()    # Check and make image saving directory
        self._draw_lineplot(data=line_plot, x='day', y='make_time')     # Draw line plot

    def draw_graph_box(self, box_plot: dict, x: str, y: str, figsize: tuple, save_nm: str) -> None:
        self._chk_and_make_dir()    # Check and make image saving directory
        self._draw_boxplot_with_point(data=box_plot, x=x, y=y, figsize=figsize,
                                      save_nm=save_nm)    # Draw box plot
        # self._draw_boxplot(data=mold_time_splitted, x='mold_id', y='make_time')

    def _calc_make_time(self, df: pd.DataFrame, name: str) -> pd.DataFrame:

        # Calculate Making times
        df[name] = df[self.date_cols[1]] - df[self.date_cols[0]]
        df[name] = df[name] / timedelta(hours=1)
        df[name] = np.round(df[name].to_numpy(), 2)       # Round

        # Work start date to date
        df['date'] = df[self.date_cols[0]].dt.date

        return df

    @staticmethod
    def _load_data(path: str, header=0, delimiter=None) -> pd.DataFrame:
        df = pd.read_csv(path, header=header, delimiter=delimiter)
        df.columns = [col.lower() for col in df.columns]    # Convert columns to lowercase

        return df

    def _conv_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.date_cols:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M')

        return df

    def _slice_day(self, df: pd.DataFrame) -> pd.DataFrame:
        # Slicing
        df = df[self.subset_col_work]
        day_hour = timedelta(hours=24)

        df_sliced = defaultdict(list)
        date_format = '%Y/%m/%d'
        for order, mold, start_dt, end_dt in zip(df['order_no'], df['mold_id'], df['work_sdt'], df['work_fdt']):
            date_range = pd.date_range(start=start_dt.strftime(date_format), end=end_dt.strftime(date_format))  # days
            date_len = len(date_range)
            make_time_fst = (day_hour - timedelta(hours=start_dt.hour, minutes=start_dt.minute))
            make_time_lst = (timedelta(hours=end_dt.hour, minutes=end_dt.minute))

            if date_len > 2:
                make_time = [make_time_fst / timedelta(hours=1)]
                make_time.extend([24] * (date_len-2))
                make_time.append(make_time_lst / timedelta(hours=1))

            elif date_len == 2:
                make_time = [make_time_fst / timedelta(hours=1), make_time_lst / timedelta(hours=1)]
            else:
                make_time = (end_dt - start_dt) / timedelta(hours=1)
                make_time = [make_time]

            df_sliced['date'].extend(date_range)
            df_sliced['order_id'].extend([order] * date_len)
            df_sliced['mold_id'].extend([mold] * date_len)
            df_sliced['make_time'].extend(make_time)

        df_sliced_df = pd.DataFrame(df_sliced)

        return df_sliced_df

    def _aggregate(self, df: pd.DataFrame, by: str, agg_col: str) -> dict:
        df_group = {}
        for agg in self.agg_list:
            group = self._group_by(df=df, by=by, agg=agg, agg_col=agg_col)
            # group.reset_index(level=0, inplace=True)
            df_group[agg] = group

        return df_group

    @staticmethod
    def _group_by(df: pd.DataFrame, by: str, agg: str, agg_col: str) -> pd.DataFrame:
        if agg == 'sum':
            df_group = df.groupby(by=by).sum()[[agg_col]]
        elif agg == 'avg':
            df_group = df.groupby(by=by).mean()[[agg_col]]
        elif agg == 'median':
            df_group = df.groupby(by=by).median()[[agg_col]]
        elif agg == 'count':
            df_group = df.groupby(by=by).count()[[agg_col]]
        else:
            raise ValueError("It's not involved in aggregation function")

        return df_group

    @staticmethod
    def _group_list(df: pd.DataFrame, key: str, val: str) -> pd.DataFrame:
        keys, values = df[[key, val]].sort_values(key).values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        group_list = pd.DataFrame({key: ukeys, val: [list(arr) for arr in arrays]})

        return group_list

    @staticmethod
    def _split_by_group(df: pd.DataFrame, group: list, split_col: str) -> dict:
        split_df = {}
        for i, group in enumerate(group):
            if isinstance(group, str):
                split_df[i] = df[df[split_col].str[:len(group)] == group]
            elif isinstance(group, list):
                temp = pd.DataFrame()
                for sub_group in group:
                    split = df[df[split_col].str[:len(sub_group)] == sub_group]
                    temp = pd.concat([temp, split], axis=0)
                split_df[i] = temp
            else:
                ValueError()

        return split_df

    @staticmethod
    def _chk_and_make_dir() -> None:
        path_img = os.path.join('..', 'img')
        if not os.path.exists(path_img):
            os.mkdir(path_img)

    def _draw_lineplot(self, data: [Dict, pd.DataFrame], x: str, y: str) -> None:
        for agg in self.agg_list:
            df = data[agg]
            ax = df.plot.line(figsize=(12, 6), linewidth=0.8, color='k')
            ax.hlines(y=df[y].mean(),
                      xmin=df.index[0], xmax=df.index[-1],
                      ls='-', color='firebrick', linewidth=0.7)
            ax.set_xlabel(x, fontsize=13)
            ax.set_ylabel(y, fontsize=13)
            ax.legend()
            plt.title(f'Making Time Trend: {agg}', fontsize=15)
            # sns.lineplot(data=df, x=x, y=y, ax=ax)
            plt.savefig(os.path.join('..', 'img', f'Line plot({agg})'))
            plt.close()

    @staticmethod
    def _draw_boxplot(data, x: str, y: str) -> None:
        for i, group in data.items():
            temp = pd.DataFrame()
            for key, val in zip(group[x], group[y]):
                temp = pd.concat([temp, pd.DataFrame({x: key, y: val})])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=temp[x], y=temp[y], ax=ax, linewidth=0.7)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.savefig(os.path.join('..', 'img', f'Mold Group {i}'))
            plt.close()

    @staticmethod
    def _draw_boxplot_with_point(data, x: str, y: str, figsize: tuple, save_nm: str) -> None:
        for i, group in data.items():
            # Merge dataset
            temp = pd.DataFrame()
            for key, val in zip(group[x], group[y]):
                temp = pd.concat([temp, pd.DataFrame({x: key, y: val})])
            fig, ax = plt.subplots(figsize=figsize)
            xvals = np.unique(temp[x])
            positions = range(len(xvals))

            # Box plot
            plt.boxplot([temp[temp[x] == xi][y] for xi in xvals],
                        positions=positions, showfliers=False,
                        boxprops={'facecolor': 'none'}, meanprops={'color': 'navy'},
                        patch_artist=True)

            # Average Line
            means = [np.mean(temp[temp[x] == xi][y]) for xi in xvals]
            plt.plot(positions, means, '--d', linewidth=0.6, color='darkred')

            # Swarm plot
            sns.swarmplot(x, y, data=temp, s=3, alpha=0.8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            plt.savefig(os.path.join('..', 'img', f'{save_nm}_{i}'))
            plt.close()

    @staticmethod
    def _re_grp_mold_type(df: pd.DataFrame, name: str) -> pd.DataFrame:
        conditions = [
            df['mold_id'].str[0] == '6',
            df['mold_id'].str[0] == '7',
            df['mold_id'].str[0] == '8',
        ]
        values = ['dual_body', 'dual_body', df['draw_id']]
        df[name] = np.select(conditions, values)

        return df


def main():
    analysis = DataAnalysis()
    # Load Dataset
    work, mold = analysis.load_dataset()

    # Data Preprocessing
    make_time_day, make_time_mold_id, make_time_mold_group = analysis.data_prep(work=work, mold=mold)

    # Draw Graph
    print("Process: Drawing Graph")
    analysis.draw_graph_line(line_plot=make_time_day)
    analysis.draw_graph_box(box_plot=make_time_mold_id, x='mold_id', y='make_time',
                            figsize=(60, 6), save_nm='mold_id')
    analysis.draw_graph_box(box_plot=make_time_mold_group, x='mold_group', y='make_time',
                            figsize=(10, 6), save_nm='mold_group')
    print("Data Analysis is finished")


main()

