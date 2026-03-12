import os
import pandas as pd
import math
import statistics
import sys
import matplotlib.pyplot as plt
import numpy as np

models_with_failure_df = pd.read_csv("./models_with_failure.csv")
all_models_with_failure = models_with_failure_df.model.tolist()
all_models_df = pd.read_csv("./data_2023/2023-12-31.csv")
all_models = all_models_df.model.tolist()
all_models.extend(all_models_with_failure)

proportions = {item: all_models_with_failure.count(item) / all_models.count(item) * 100 for item in set(all_models_with_failure)}

sorted_proportions = dict(sorted(proportions.items(), key=lambda item: item[1], reverse=True))


labels = list(sorted_proportions.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

plt.figure(figsize=(15,15))
bars = plt.bar(sorted_proportions.keys(), sorted_proportions.values(), color=colors)
plt.xlabel('model', fontsize=1)
plt.ylabel('Failure percentage ($)')
plt.title('Proportional failure percentage per disk model (#disks per model failed / #disks per model installed in the datacenters)')
plt.xticks(rotation=45, ha='right')

legend_labels = [key for key in sorted_proportions.keys()]
plt.legend(bars, legend_labels, fontsize=6)

max_value = max(sorted_proportions.values())
max_label = [label for label, value in sorted_proportions.items() if value == max_value][0]

# plt.gca().get_xticklabels()[list(sorted_proportions.keys()).index(max_label)].set_fontweight('bold')

plt.show()

print(f'Total disks with model {max_label} installed: {all_models.count(max_label)}')
print(f'Total disks with model {max_label} failed: {all_models_with_failure.count(max_label)}')