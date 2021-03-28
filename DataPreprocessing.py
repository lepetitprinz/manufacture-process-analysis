import os
import warnings
import datetime as dt

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class DataPreprocessing(object):
    # Rename columns
    RN_WORK = {'번호': 'number', '금형번호': 'mold_id', '공정명': 'proc_nm', 'TRY': 'try_out',
               '완료여부': 'status', '작업수량': 'mnt', '작업자': 'worker_nm', '작업내용': 'work_info',
               '작업일': 'work_day', '작업시간': 'work_time'}
    RN_WORK_TIME = {'근태일자': 'work_day', '부서': 'dep', '성명': 'worker_nm', '사번': 'worker_id',
                    '근태구분': 'work_type', '출근시간': 'get_to_time', '퇴근시간': 'get_off_time',
                    '철야근무일수(4hr/일)': 'over_4_yn', '철야근무일수(6hr/일)': 'over_6_yn', '야간근무일수': 'night_yn'}

    # Drop columns
    DROP_COL_WORK = ['number']
    DROP_COL_WORK_TIME = ['dep', 'worker_id', 'work_type']

    def __init__(self):
        self.path_data = os.path.join('..', 'data')
        self.path_save = os.path.join('..', 'result')

    def run(self):
        # Load dataset
        work = pd.read_csv(os.path.join(self.path_data, 'work.csv'), delimiter='\t')
        work_time = pd.read_csv(os.path.join(self.path_data, 'work_time.csv'), delimiter='\t')

        # Rename columns
        work = work.rename(columns=self.__class__.RN_WORK)
        work_time = work_time.rename(columns=self.__class__.RN_WORK_TIME)

        # Drop columns
        work = work.drop(columns=self.__class__.DROP_COL_WORK)
        work_time = work_time.drop(columns=self.__class__.DROP_COL_WORK_TIME)

        # Add columns
        work_time['over_yn'] = work_time['over_4_yn'] + work_time['over_6_yn'] + work_time['night_yn']
        work_time = work_time.drop(columns=['over_4_yn', 'over_6_yn', 'night_yn'])

        # Time preprocessing
        work['make_time_end'] = work['work_day'] + ' ' + work['work_time']
        work['make_time_end'] = pd.to_datetime(work['make_time_end'], format='%Y-%m-%d %H:%M:%S')

        work_time = self.conv_work_time_format(work_time=work_time)

        # Remove unnecessary columns
        work = work.drop(columns=['work_time'])
        work_time = work_time.drop(columns=['get_to_time', 'get_off_time', 'over_yn'])

        # Merge dataset
        merged = pd.merge(work, work_time, how='left', on=['work_day', 'worker_nm'],
                          left_index=True, right_index=False)

        # Sort dataset
        merged = merged.sort_values(by=['work_day', 'worker_nm', 'make_time_end'])

        # Remove NA data
        merged = merged.dropna(axis=0, subset=['make_time_end'])

        #
        worker_list = merged['worker_nm'].unique()

        merged_worker = pd.DataFrame()
        for worker in worker_list:
            df_by_worker = merged[merged['worker_nm'] == worker]
            make_time_end_bf = list(df_by_worker['make_time_end'].to_numpy()[:-1])
            make_time_end_bf.insert(0, dt.datetime(2020, 1, 1))
            df_by_worker['make_time_end_bf'] = make_time_end_bf

            # Exception
            df_by_worker['make_time_end_day'] = df_by_worker['make_time_end'].dt.strftime('%Y-%m-%d')
            df_by_worker['make_time_end_bf_day'] = df_by_worker['make_time_end_bf'].dt.strftime('%Y-%m-%d')

            temp = np.where(df_by_worker['make_time_end_day'] != df_by_worker['make_time_end_bf_day'],
                            df_by_worker['work_time_start'], df_by_worker['make_time_end_bf'])
            df_by_worker['make_time_end_bf'] = temp

            merged_worker = pd.concat([merged_worker, df_by_worker], axis=0)

        merged_worker = merged_worker.drop(columns=['make_time_end_day', 'make_time_end_bf_day'])

        # Save dataset
        merged_worker.to_csv(os.path.join(self.path_save, 'result_by_worker.csv'), index=False, sep='\t',
                             encoding='utf-8')

        merged_mold = merged_worker.drop(columns=['work_day', 'work_time_start', 'work_time_end', 'status'])
        merged_mold = merged_mold[['mold_id', 'make_time_end_bf', 'make_time_end', 'proc_nm',
                                   'try_out', 'mnt', 'work_info']]
        merged_mold = merged_mold.sort_values(by=['mold_id', 'make_time_end_bf', 'make_time_end'])

        merged_mold.to_csv(os.path.join(self.path_save, 'result_by_mold.csv'), index=False, sep='\t',
                           encoding='utf-8')

        print("Data preprocessing is finished")

    @ staticmethod
    def conv_work_time_format(work_time: pd.DataFrame):
        # Start time
        work_time['work_time_start'] = work_time['work_day'] + ' ' + work_time['get_to_time']
        work_time['work_time_start'] = pd.to_datetime(work_time['work_time_start'], format='%Y-%m-%d %H:%M')

        # End time
        work_time['work_time_end'] = work_time['work_day'] + ' ' + work_time['get_off_time']
        work_time['work_time_end'] = pd.to_datetime(work_time['work_time_end'], format='%Y-%m-%d %H:%M')

        # Add 1 day on (over & night time)
        work_time['work_time_end'] = np.where(work_time['over_yn'] == 0, work_time['work_time_end'],
                                              work_time['work_time_end'] + dt.timedelta(days=1))

        # Exception (work start time > work end time)
        work_time['work_time_end'] = np.where(work_time['work_time_start'] > work_time['work_time_end'],
                                              work_time['work_time_end'] + dt.timedelta(days=1),
                                              work_time['work_time_end'])

        return work_time


def main():
    prep = DataPreprocessing()
    prep.run()
    print("")


main()