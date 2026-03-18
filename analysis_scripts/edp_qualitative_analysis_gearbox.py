# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils import loadDataset
import matplotlib.pyplot as plt
import pandas as pd

dataset = loadDataset.get_dataset("edp-wt")

print()

index_of_turbine_of_interest = dataset['target_sources'].index('T06')
turbine_of_interest_data = dataset['target_data'][index_of_turbine_of_interest]
failure_occurence_index = 41485

turbine_of_interest_data = turbine_of_interest_data.loc[:failure_occurence_index]

gear_oil_temp_avg = turbine_of_interest_data['Gear_Oil_Temp_Avg']
gear_bear_temp_avg = turbine_of_interest_data['Gear_Bear_Temp_Avg']

min_value = min(gear_oil_temp_avg.min(), gear_bear_temp_avg.min())
max_value = max(gear_oil_temp_avg.max(), gear_bear_temp_avg.max())

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18,10), sharex=True)

axes[0].plot(gear_oil_temp_avg, label='Gear Oil Temp Avg', linewidth=1, color='blue')
axes[1].plot(gear_bear_temp_avg, label='Gear Bear Temp Avg', linewidth=1, color='green')

# axes[0].ylim(ymin=min_value-1, ymax=max_value+1)
# axes[1].ylim(ymin=min_value-1, ymax=max_value+1)

for ax in axes[:2]:
    ax.fill_between(turbine_of_interest_data.index[-289:-1], min_value-1, [max_value + 1 for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
    ax.fill_between(turbine_of_interest_data.index[-8929:-289], min_value-1, [max_value + 1 for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
    ax.axvline(x=failure_occurence_index, color='red', linewidth=3)

pb_scores = pd.read_csv('ProfileBased.csv')
axes[2].plot(pb_scores['0'], label='Profile based', linewidth=1, color='purple')
axes[2].fill_between(turbine_of_interest_data.index[-289:-1], pb_scores['0'].min(), [pb_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
axes[2].fill_between(turbine_of_interest_data.index[-8929:-289], pb_scores['0'].min(), [pb_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
axes[2].axvline(x=failure_occurence_index, color='red', linewidth=4)

ltsf_scores = pd.read_csv('LTSFLinear.csv')
axes[3].plot(ltsf_scores['0'], label='LTSF', linewidth=1, color='black')
axes[3].fill_between(turbine_of_interest_data.index[-289:-1], ltsf_scores['0'].min(), [ltsf_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
axes[3].fill_between(turbine_of_interest_data.index[-8929:-289], ltsf_scores['0'].min(), [ltsf_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
axes[3].axvline(x=failure_occurence_index, color='red', linewidth=4)

handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)



fig.legend(handles, labels, loc='upper center', ncol=2)

plt.legend()
plt.show()
# 41485