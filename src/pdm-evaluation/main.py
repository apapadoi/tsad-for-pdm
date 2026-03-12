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
