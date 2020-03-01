import os
import cv2
import pandas as pd
from datetime import datetime, timedelta

import json
import warnings
from bokeh.util.warnings import BokehUserWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from bokeh.layouts import  column
from bokeh.models import CustomJS, Slider
from bokeh.plotting import figure, output_file, ColumnDataSource, save


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
    df['exp_time'] = df['exp_time'] / 1000
    df['sysdatetime'] = datetime.strptime(dic['date'] + '_' + dic['time'], '%Y-%m-%d_%H:%M:%S.%f')  # .time()
    df['sysdatetime'] = df['sysdatetime'] + df['exp_time'].apply(lambda x: timedelta(seconds=x - 3600 * 3))
    return df, dic


def get_glasses_df(glasses_path, N=3000):

    info_path = glasses_path + '/info.player.json'
    if os.path.isfile(info_path):
        #data from desktop
        video_path = glasses_path + '/world.mp4'
        with open(info_path) as f:
            dic = json.loads(f.read())
            start_time_system_s = dic['start_time_system_s']
    else:
        #data from mobile
        info_path = glasses_path + '/info.csv'
        video_path = glasses_path + '/Pupil Cam1 ID2.mp4'
        with open(info_path) as f:
            data =  f.read()
            dic = dict(list(map(lambda x: x.split(','), data.split('\n')[1:-1])))
            start_time_system_s = dic['Start Time (System)']
    cap = cv2.VideoCapture(video_path)
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
    df['exp_time'] = df['exp_time'] / 1000
    df['sysdatetime'] = df['exp_time'].apply(
        lambda x: timedelta(seconds=x) + datetime.utcfromtimestamp(float(start_time_system_s)))
    return df, dic


def get_tablet_df(tablet_path, N=10000):
    df = pd.read_csv(tablet_path)
    df['sync_col'] = df['illuminance (lx)'] / df['illuminance (lx)'].max()
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


def build_graphs(dic_df, base_name, sync_names, experiment_name, output_path):
    warnings.filterwarnings("ignore", category=BokehUserWarning)
    js_pattern = """
        const data = source.data;
        {}
        for (var i = 0; i < data['xs'][1].length; i++) {
            data['xs'][1][i] = data['xs2'][1][i]+phase.value+phase2.value;
        }
        source.change.emit();
    """

    for sync_name in sync_names:

        delta = dic_df[sync_name]['sysdatetime'].iloc[0] - dic_df[base_name]['sysdatetime'].iloc[0]
        delta = delta.total_seconds()

        source_dict = dict(
            xs=[dic_df[base_name]['exp_time'].values, dic_df[sync_name]['exp_time'].values+delta],
            ys=[dic_df[base_name]['sync_col'].values, dic_df[sync_name]['sync_col'].values],
            line_color=['red', 'blue'] * 2
        )

        source_dict['xs2'] = [dic_df[base_name]['exp_time'].values, dic_df[sync_name]['exp_time'].values]

        source_dict['legend_label'] = [base_name, sync_name] * 2
        source_dict['line_width'] = [2] * 2 + [0] * 2

        p = figure(width=1500, height=500)
        source = ColumnDataSource(source_dict)


        phase_slider = Slider(start=delta - 10, end=delta + 10, value=delta, step=.001, title="Phase")
        phase_slider2 = Slider(start=-1, end=1, value=0, step=.001, title="Phase2")

        callback = CustomJS(args=dict(source=source, phase=phase_slider, phase2=phase_slider2),
                            code=js_pattern.replace('{}', 'phase2.value=0;'))
        callback2 = CustomJS(args=dict(source=source, phase=phase_slider, phase2=phase_slider2),
                             code=js_pattern.replace('{}', ''))

        phase_slider.js_on_change('value', callback)
        phase_slider2.js_on_change('value', callback2)

        p.multi_line(xs="xs", ys="ys", source=source,
                     line_color='line_color',
                     legend_label='legend_label',
                     line_width='line_width'
                     )

        layout = column(p, phase_slider)
        output_file(f'{output_path}/{experiment_name}_{base_name}_{sync_name}.html')
        save(layout)
