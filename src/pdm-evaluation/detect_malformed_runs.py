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

import os
import yaml
filestore_root_dir = "./mlruns" # insert your filestore dir (string) here
experiment_id = 0 # insert your experiment ID (int) here
experiment_id_list = os.listdir(filestore_root_dir)
experiment_id_list.remove('0')
experiment_id_list.remove('.trash')
print(len(experiment_id_list))

for experiment_id in experiment_id_list:
    experiment_dir = os.path.join(filestore_root_dir, str(experiment_id))
    for run_dir in [elem for elem in os.listdir(experiment_dir) if elem != "meta.yaml"]:
      meta_file_path = os.path.join(experiment_dir, run_dir, 'meta.yaml')
      with open(meta_file_path) as meta_file:
        if yaml.safe_load(meta_file.read()) is None:
          print("Run data in file %s was malformed" % meta_file_path)