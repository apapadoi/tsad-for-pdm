import sys

from utils import loadDataset
import matplotlib.pyplot as plt
import pandas as pd

dataset = loadDataset.get_dataset("edp-wt")


index_of_turbine_of_interest = dataset['target_sources'].index('T07')
turbine_of_interest_data = dataset['target_data'][index_of_turbine_of_interest]
pause_pressed_on_keyboard = 33285
episode_start_index = 24101 # 0
failure_occurence_index = 33307

turbine_of_interest_data = turbine_of_interest_data.loc[episode_start_index:failure_occurence_index]

channels_of_interest = [
    'Gen_RPM_Max',
    'Gen_RPM_Min',
    'Gen_RPM_Avg',
    'Gen_RPM_Std',
    'Gen_Bear_Temp_Avg',
    'Gen_Phase1_Temp_Avg',
    'Gen_Phase2_Temp_Avg',
    'Gen_Phase3_Temp_Avg',
    'Prod_LatestAvg_ActPwrGen0',
    'Prod_LatestAvg_ActPwrGen1',
    # 'Prod_LatestAvg_ActPwrGen2', # constant column, removed in loadDataset
    'Prod_LatestAvg_ReactPwrGen0',
    'Prod_LatestAvg_ReactPwrGen1',
    # 'Prod_LatestAvg_ReactPwrGen2', # constant column, removed in loadDataset
    'Gen_SlipRing_Temp_Avg',
    'Gen_Bear2_Temp_Avg'
]

# plt.figure(figsize=(18, 10))

# for channel in channels_of_interest:
#     s = turbine_of_interest_data[channel]
#     plt.plot((s - s.min()) / (s.max() - s.min()), label=channel, linewidth=1)
# gear_oil_temp_avg = turbine_of_interest_data['Gear_Oil_Temp_Avg']
# gear_bear_temp_avg = turbine_of_interest_data['Gear_Bear_Temp_Avg']

min_value = sys.float_info.max
max_value = sys.float_info.min
for channel in channels_of_interest:
    s = turbine_of_interest_data[channel].copy()
    s = (s - s.min()) / (s.max() - s.min())

    if s.max() > max_value:
        max_value = s.max()

    if s.min() < min_value:
        min_value = s.min()

# min_value = min(gear_oil_temp_avg.min(), gear_bear_temp_avg.min())
# max_value = max(gear_oil_temp_avg.max(), gear_bear_temp_avg.max())

fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(100,35), sharex=True)
#
# axes[0].plot(gear_oil_temp_avg, label='Gear Oil Temp Avg', linewidth=1, color='blue')
# axes[1].plot(gear_bear_temp_avg, label='Gear Bear Temp Avg', linewidth=1, color='green')
#
# # axes[0].ylim(ymin=min_value-1, ymax=max_value+1)
# # axes[1].ylim(ymin=min_value-1, ymax=max_value+1)
#
# for ax in axes[:2]:
FONT_SIZE = 65
cmap = plt.get_cmap('tab20_r')
for index in range(7):
    ax = axes[index][0]
    current_channel = channels_of_interest[index]
    ax.plot(turbine_of_interest_data[current_channel], label=current_channel.replace('_', ' '), linewidth=2, color=cmap(index))
    ax.fill_between(turbine_of_interest_data.index[-289:-1], turbine_of_interest_data[current_channel].min(),
                                 [turbine_of_interest_data[current_channel].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))],
                                 color='grey', alpha=0.3)
    ax.fill_between(turbine_of_interest_data.index[-8929:-289], turbine_of_interest_data[current_channel].min(),
                                 [turbine_of_interest_data[current_channel].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))],
                                 color='red', alpha=0.2)
    ax.axvline(x=failure_occurence_index, color='red', linewidth=2)
    ax.axvline(x=pause_pressed_on_keyboard, color='green', linewidth=2)


for index in range(7):
    ax = axes[index][1]
    current_channel = channels_of_interest[index+7]
    ax.plot(turbine_of_interest_data[current_channel], label=current_channel.replace('_', ' '), linewidth=2, color=cmap(index+7))
    ax.fill_between(turbine_of_interest_data.index[-289:-1], turbine_of_interest_data[current_channel].min(),
                                 [turbine_of_interest_data[current_channel].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))],
                                 color='grey', alpha=0.3)
    ax.fill_between(turbine_of_interest_data.index[-8929:-289], turbine_of_interest_data[current_channel].min(),
                                 [turbine_of_interest_data[current_channel].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))],
                                 color='red', alpha=0.2)

    if index == 0:
        ax.axvline(x=failure_occurence_index, color='red', linewidth=2, label='Generator damaged')
        ax.axvline(x=pause_pressed_on_keyboard, color='green', linewidth=2, label='Pause pressed on keyboard')
    else:
        ax.axvline(x=failure_occurence_index, color='red', linewidth=2)
        ax.axvline(x=pause_pressed_on_keyboard, color='green', linewidth=2)


pb_scores = pd.read_csv('T07_ProfileBased.csv')[episode_start_index:failure_occurence_index]
ltsf_scores = pd.read_csv('T07_LTSFLinear.csv')[episode_start_index:failure_occurence_index]

for column in range(2):
    axes[7][column].plot(pb_scores['0'], label='PB', linewidth=1, color='purple')
    axes[7][column].fill_between(turbine_of_interest_data.index[-289:-1], pb_scores['0'].min(), [pb_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
    axes[7][column].fill_between(turbine_of_interest_data.index[-8929:-289], pb_scores['0'].min(), [pb_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
    axes[7][column].axvline(x=failure_occurence_index, color='red', linewidth=2)
    axes[7][column].axvline(x=pause_pressed_on_keyboard, color='green', linewidth=2)


    axes[8][column].plot(ltsf_scores['0'], label='LTSF', linewidth=1, color='black')
    axes[8][column].fill_between(turbine_of_interest_data.index[-289:-1], ltsf_scores['0'].min(), [ltsf_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
    axes[8][column].fill_between(turbine_of_interest_data.index[-8929:-289], ltsf_scores['0'].min(), [ltsf_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
    axes[8][column].axvline(x=failure_occurence_index, color='red', linewidth=2)
    axes[8][column].axvline(x=pause_pressed_on_keyboard, color='green', linewidth=2)
#
handles, labels = [], []
for column in range(2):
    for row in range(9):
        h, l = axes[row][column].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

fig.legend(handles, labels, loc='upper center', ncol=9, fontsize=FONT_SIZE - 30)


plt.legend()
plt.show()
