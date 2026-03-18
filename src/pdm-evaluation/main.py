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

from experimental_runs_configuration.ims.run_unsupervised import execute as execute_ims_unsupervised

methods_to_run_unsupervised = [
        'KNN',
        'IF',
        'LOF',
        'NP',
        'SAND'
]

MAX_JOBS = 12
MAX_RUNS = 50
INITIAL_RANDOM = 2

execute_ims_unsupervised(
    method_names_to_run=methods_to_run_unsupervised, 
    MAX_RUNS=MAX_RUNS, 
    MAX_JOBS=MAX_JOBS, 
    INITIAL_RANDOM=INITIAL_RANDOM
)
