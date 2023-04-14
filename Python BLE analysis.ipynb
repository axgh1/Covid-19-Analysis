import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import re
import os
import glob

import sys
sys.path.append('../')
from scib import Plotter

plt.rcParams["font.size"] = 18
log_categories = {
    'adv': {
        'columns': ['time', 'addr'],
        'dtypes': [int, str],
        'pattern': r'00> (\d{10}),a,([0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2})*'
    },
    'connected': {
        'columns': ['time', 'conn_handle', 'addr'],
        'dtypes': [int, int, str],
        'pattern': r'00> (\d{10}),c,(\d+),([0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2})*'
    },
    'disconnected': {
        'columns': ['time', 'cmu_index', 'addr'],
        'dtypes': [int, int, str],
        'pattern': r'00> (\d{10}),d,(\d+),([0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2}:[0-9a-zA-Z]{2})*'
    },
    'vs': {
        'columns': ['time', 'conn_handle', 'cc'],
        'dtypes': [int, int, int],
        'pattern': r'00> (\d{10}),v,(\d+),(\d+$)'
    },
    'qos': {
        'columns': ['time', 'conn_handle', 'channel_index', 'event_counter', 'crc_ok_count', 'crc_error_count', 'nak_count', 'rx_timeout', 'rssi'],
        'dtypes': [int, int, int, int, int, int, int, int, int],
        'pattern': r'00> (\d{10}),q,(\d+),(\d+),(\d{6}),(\d+),(\d+),(\d+),(\d+),(-\d+$)'
    }
}
log_dir = 'log'
log_files = {
    'g1': '20230413-1544-bmu.log',
    # 'g2': '20230214_183443_COM42.log',
    # 'g3': '20230214_183508_COM56.log',
    # 'g4': '20230214_183512_COM57.log'
}
dfs = {}
for group in log_files.keys():
    dfs[group] = {}
dfs_filename = os.path.join(log_dir, 'dfs.pickle')

if os.path.isfile(dfs_filename):
    with open(dfs_filename, 'rb') as f:
        dfs = pickle.load(f)
        print('dfs has been loaded.')
else:
    for group, filename in log_files.items():
        print(group)
        with open(os.path.join(log_dir, filename)) as f:
            lines = f.readlines()
            for df_name, info in log_categories.items():
                repatter = re.compile(info['pattern'])
                match_lines = [repatter.match(l).groups() for l in lines if repatter.match(l)]
                dfs[group][df_name] = pd.DataFrame(match_lines, columns=info['columns'], dtype=str)
                for column, dtype in zip(info['columns'], info['dtypes']):
                    dfs[group][df_name][column] = dfs[group][df_name][column].astype(dtype)
                dfs[group][df_name]['timedelta'] = dfs[group][df_name]['time'].apply(lambda x: datetime.timedelta(milliseconds=x))
                dfs[group][df_name].set_index('timedelta')
    with open(dfs_filename, 'wb') as f:
        pickle.dump(dfs, f)
        print('dfs has been dumped.')
# Check the duration in hours
for group in dfs.keys():
    print(group, datetime.timedelta(milliseconds=dfs[group]['vs']['time'].max().item()))
n_cmu = 0
for group in dfs.keys():
    n_cmu = max(n_cmu, len(dfs[group]['connected']['conn_handle'].unique()))
n_cmu
dfs['g1']['qos']
dfs['g1']['qos']

group_handle2addr = {}
for group in dfs.keys():
    group_handle2addr[group] = {}
    for conn_handle in range(n_cmu):
        group_handle2addr[group][conn_handle] = dfs[group]['connected'].query('conn_handle==@conn_handle').iloc[-1]['addr']
group_handle2addr
datetime.timedelta(minutes=30).total_seconds() * 1000
# Remove first 30 minutes and leave 8 hours data
# for group, df_group in dfs.items():
#     for category, df in df_group.items():
#         time_start = datetime.timedelta(minutes=30).total_seconds() * 1000
#         time_end = time_start + datetime.timedelta(hours=8).total_seconds() * 1000
#         dfs[group][category] = df.query('@time_start < time < @time_end').reset_index(drop=True)
# Check the duration in hours
for group in dfs.keys():
    print(group, datetime.timedelta(milliseconds=dfs[group]['vs']['time'].max().item()))
# Check the number of conn handle
for group in dfs.keys():
    print(group, len(dfs[group]['vs']['conn_handle'].unique()))
for group in dfs.keys():
    for conn_handle in range(n_cmu):
        dfs[group]['vs'].loc[dfs[group]['vs']['conn_handle']==conn_handle, 'delayed'] = dfs[group]['vs'].query('conn_handle==@conn_handle')['time'].diff()>150
        dfs[group]['vs'].loc[dfs[group]['vs']['conn_handle']==conn_handle, 'delayed'] = dfs[group]['vs'].loc[dfs[group]['vs']['conn_handle']==conn_handle, 'delayed'].astype(float)
# Check the number of disconnection
for group in dfs.keys():
    print(group, len(dfs[group]['disconnected']))
## Analysis 


dfs['g1']['qos']['channel_index'].max()
dfs['g1']['qos'][['crc_ok_count', 'crc_error_count']].drop_duplicates()
dfs['g1']['qos']
dfs['g1']['qos']['crc_error_count'].sum()
dfs['g1']['qos'].query('channel_index==0')
for i in range(3):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.show()
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot()
df=dfs['g1']['qos'].query('channel_index==3')
x=df['time']
y=df['rssi']
ax.plot(x, y)
for group in dfs.keys():
    dfs[group]['qos']['result'] = pd.Series(dtype=str)
    dfs[group]['qos'].loc[(dfs[group]['qos']['crc_ok_count']==0) & (dfs[group]['qos']['crc_error_count']==0), 'result'] = 'not arrived'
    dfs[group]['qos'].loc[(dfs[group]['qos']['crc_ok_count']>0), 'result'] = 'ok'
    dfs[group]['qos'].loc[(dfs[group]['qos']['crc_error_count']>0), 'result'] = 'error'
# delay check
for group, df_group in dfs.items():
    for conn_handle, addr in sorted(group_handle2addr[group].items(), key=lambda x: x[1]):
        print(group, end='\t')
        print(conn_handle, end='\t')
        df_diff = df_group['vs'].query('conn_handle==@conn_handle')['time'].diff().iloc[1:, ]
        print(f'max delay:{df_diff.max()}(ms)', end='\t')
        # print(f'delay prob:{len(df_diff[df_diff>150])/len(df_diff):.5e}')
        df_diff_qos = df_group['qos'].query('conn_handle==@conn_handle')['time'].diff().iloc[1:, ]
        print(f'max delay(qos):{df_diff_qos.max()}(ms)', end='\t')

        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(f'{group}-{addr}-conn handle({conn_handle})')
        ax1 = fig.add_subplot(2, 2, 1)
        Plotter.ecdf(df_group['vs'], conn_handle, ax1)

        ax2 = fig.add_subplot(2, 2, 2)
        Plotter.rssi_box(df_group['qos'], conn_handle, ax2)

        ax3 = fig.add_subplot(2,2,3)
        Plotter.delay_transition_qos(df_group['qos'], conn_handle, ax3, ylim=(0, 100))
        # Plotter.delay_transition(df_group['vs'], conn_handle, ax3)

        ax4 = fig.add_subplot(2, 2, 4)
        Plotter.crc_hist(df_group['qos'], conn_handle, ax4)

        plt.tight_layout()
        plt.show()

        list(df_group['qos'][df_group['qos']['conn_handle'] == conn_handle]['channel_index'].unique())
        print()
# Group data by channel number
grouped_data = dfs['g1']['qos'].groupby('channel_index')

# Plot RSSI by time for each channel
for channel, group in grouped_data:
    plt.plot(group['time'], group['rssi'])
    plt.title('RSSI by Time for Channel {}'.format(channel))
    plt.xlabel('Time')
    plt.ylabel('RSSI')
    plt.show()
# Group data by channel number
grouped_data = dfs['g1']['qos'].groupby('channel_index')

# Plot ECDF for each channel
for channel, group in grouped_data:
    # Calculate the time intervals between packets
    intervals = group['timedelta'].diff().dropna()

    # Plot the ECDF using seaborn.ecdfplot
    sns.ecdfplot(intervals, label=f'Channel {channel}')

    # Set axis labels and legend
    plt.xlabel('Packet Interval (ms)')
    plt.ylabel('CDF')
    plt.legend()
    plt.title('CDF by Interval for Channel {}'.format(channel))

    # Show plot
    plt.show()
# Group data by connection handle
grouped_data = dfs['g1']['vs'].groupby('conn_handle')

# Plot ECDF for each connection handle
for connectionhandle, group in grouped_data:
    # Calculate the time intervals between packets
    intervals = group['timedelta'].diff().dropna()

    # Plot the ECDF using seaborn.ecdfplot
    sns.ecdfplot(intervals, label=f'Connection handle {connectionhandle}')

    # Set axis labels and legend
    plt.xlabel('Packet Interval (ms)')
    plt.ylabel('CDF')
    plt.legend()
    plt.title('CDF by Interval')

    # Show plot
    plt.show()
dfs['g1']['vs']
## Delay prob
def over360(p):
    n=6
    P = np.array(
        [([0, 1.] + [0 for i in range(n-2)])]+[[1-p,0]+[p if i==j else 0 for j in range(n-2)] for i in range(n-2)]+[([0 for i in range(n-1)] + [1])],
        dtype='float64'
    ).T
    v_init = np.array(
        [1] + [0 for i in range(n-1)],
        dtype='float64'
    )
    n_event = int(datetime.timedelta(days=365*10).total_seconds() / 0.06)
    return ((np.linalg.matrix_power(P, n_event))@v_init)[-1]
(dfs['g1']['qos']['crc_error_count'] + dfs['g1']['qos']['crc_ok_count']).idxmax()
pd.set_option('display.max_columns', 100)
_df = dfs['g4']['qos']
_df.loc[_df['crc_error_count']+_df['crc_ok_count']>=2].T
worst_p = -1
worst_group = ''
worst_addr = ''
for group, df_group in dfs.items():
    for conn_handle, addr in sorted(group_handle2addr[group].items(), key=lambda x: x[1]):
        df_qos = df_group['qos'].query('conn_handle==@conn_handle')
        # s = df_group((df_group['qos']['conn_handle']==conn_handle)&(df_group['qos']['result']!='ok'))
        s = (df_qos['result']!='ok')
        p = s.sum() / s.count()
        worst_p = max(p, worst_p)
        if p==worst_p:
            worst_group = group
            worst_addr = addr
        print(f'group:{group}, conn handle:{conn_handle}, addr: {addr}, p: {p:.3e}, p(over360):{over360(p):.3e}')

print(worst_group, worst_addr)
print(f'worst p:{worst_p:.3e}, over360: {over360(worst_p):.3e}')
