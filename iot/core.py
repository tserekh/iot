import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cv2
import json

def get_bitalino_df(bitalino_path: str, N: int, lux_colname='A6', sync_col='sync_col'):
    f = open(bitalino_path)
    lines = f.readlines()[:10]

    import json

    dic = json.loads(lines[1].replace('#', '', 1))
    dic = list(dic.values())[0]

    df = pd.read_csv(bitalino_path, skiprows=3, header=None, encoding='cp1251', sep='\t', names=dic['column'] + [''],
                     nrows=N)
    df = df.drop('', axis=1)

    df[lux_colname] = df[lux_colname].astype(float)
    max_val = df[lux_colname].max()
    if max_val == 0:
        max_val = 1
    df[sync_col] = df[lux_colname] / max_val

    df[sync_col] = df[lux_colname].astype(float) / df[lux_colname].astype(float).max()
    df = df.reset_index().rename(columns={'index': 'exp_time'})
    df['exp_time'] = df['exp_time']/1000
    df['sysdatetime'] = datetime.strptime(dic['date'] + '_' + dic['time'], '%Y-%m-%d_%H:%M:%S.%f')  # .time()
    df['sysdatetime'] = df['sysdatetime'] + df['exp_time'].apply(lambda x: timedelta(seconds=x - 3600*3))
    return df, dic

def get_glasses_df(glasses_path,  N=3000):
    cap = cv2.VideoCapture(glasses_path+'/world.mp4')

    info_path = glasses_path + '/info.player.json'
    with open(info_path) as f:
        dic = json.loads(f.read())

    ok = True
    brightness = []
    timestamps = []
    for _ in range(N):
        ok, image = cap.read()

        if ok:
            brightness.append(image.mean())
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not ok:
            break
    brightness = np.array(brightness)

    df = pd.DataFrame([brightness, timestamps], index=['brightness', 'exp_time']).T
    df['sync_col'] = df['brightness'] / df['brightness'].max()
    df['exp_time'] = df['exp_time']/1000
    df['sysdatetime'] = df['exp_time'].apply(
        lambda x: timedelta(seconds=x) + datetime.utcfromtimestamp(dic['start_time_system_s']))
    return df, dic

def get_tablet_df(tablet_path, N = 10000):

    df = pd.read_csv(tablet_path)
    df['sync_col'] = df['illuminance (lx)']/df['illuminance (lx)'].max()
    df['exp_time'] = (df['epoch (ms)'] - df['epoch (ms)'].min()) / 1000
    df['sysdatetime'] = df['epoch (ms)'].apply(lambda x: datetime.utcfromtimestamp(x / 1000))
    return df, {}


def get_opt_delta(df1, df2, deltas):
    cols = ['sync_col', 'exp_time', 'sysdatetime']
    df1['exp_time_round'] = np.round(df1['exp_time'], 1)
    df2['exp_time_round'] = np.round(df2['exp_time'], 1)
    errors = []
    deltas_return = []
    for delta in tqdm_notebook(deltas):
        df_sub = pd.merge(df1, df2, on='exp_time_round')

        df_sub['cumsum'] = df_sub['sync_col_x'].cumsum()
        df_sub = df_sub[df_sub['cumsum'] != df_sub['cumsum'].min()]
        df_sub = df_sub[df_sub['cumsum'] != df_sub['cumsum'].max()]

        if df_sub.shape[0]:
            error = mean_absolute_error(df_sub['sync_col_x'], df_sub['sync_col_y'])
            errors.append(error)
            deltas_return.append(delta)
        else:
            continue

        if not errors:
            return None, None
    if not errors:
        return None, None
    errors = np.array(errors)
    deltas_return = np.array(deltas_return)
    min_error = errors.min()
    delta = deltas_return[errors == min_error][0]
    return delta, min_error
