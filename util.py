import datetime
import pandas as pd


class util(object):

    df_human = pd.read_excel("C:/Users/user/wyjang/01.nanoace/00.data/근태시간_200801-210216.xlsx", header=None, skiprows=2)
    df_human_mody = df_human[[0, 1, 2, 5, 6, 12, 13, 16]]
    df_human_mody.columns = ['GWD', 'DEPT_NM', 'WORKER_NM', 'GWT', 'LWT', "OV4", "OV6", "REV"]

    def conv_datetime(df_human_mody, worker_nm, datetime):
        GWD = str(datetime)[:10]
        select1 = df_human_mody["WORKER_NM"] == worker_nm
        select2 = df_human_mody["GWD"] == GWD

        if df_human_mody[select1 & select2].empty:
            a = datetime.datetime.strptime(str(GWD) + " " + "08:00:00", "%Y-%m-%d %H:%M:%S")
            b = datetime.datetime.strptime(str(GWD) + " " + "17:00:00", "%Y-%m-%d %H:%M:%S")
        else:
            # 철야 + 야간 근무인 경우 date + 1
            a = datetime.datetime.strptime( str(GWD) + " " + list(df_human_mody[select1 & select2]["GWT"] + ":00")[0] , "%Y-%m-%d %H:%M:%S")
            b = datetime.datetime.strptime( str(GWD) + " " + list(df_human_mody[select1 & select2]["LWT"] + ":00")[0] , "%Y-%m-%d %H:%M:%S")
            c = list(df_human_mody[select1 & select2]["REV"])[0]
            d = list(df_human_mody[select1 & select2]["OV4"])[0]
            e = list(df_human_mody[select1 & select2]["OV6"])[0]
            b += datetime.timedelta(days=c + d + e)
        if a > b:
            b += datetime.timedelta(days = 1)
        GWT_LWT = (a, b)
        return GWT_LWT