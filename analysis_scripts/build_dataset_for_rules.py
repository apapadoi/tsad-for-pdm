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

import mlflow
import sys
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from mango.tuner import Tuner
from mango import scheduler


RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
mpl.rcParams['svg.hashsalt'] = str(RANDOM_STATE)

flavors = [
    "Auto profile ",
    "Incremental ",
    "Semisupervised ",
    "Unsupervised "

]
formal_flavor_name_map = {
    'Auto profile ': 'online',
    'Incremental ': 'sliding',
    'Semisupervised ': 'historical',
    'Unsupervised ': 'unsupervised'
}
FONT_SIZE = 65

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')

df.drop_duplicates(inplace=True, ignore_index=True)

df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'], axis=1)

# df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200] # include chronos for online with cmapps ph 19
df.drop(columns=['VUS_PR'], inplace=True)
df = df[df['Technique'] != 'XGBOOST']
# add dataset characteristics
# datasets = [
#     'cmapss',
#     'navarchos',
#     'femto',
#     'ims',
#     'edp-wt',
#     'metropt-3',
#     'xjtu',
#     'bhd',
#     'azure',
#     'ai4i'
# ]

dataset_characteristics_df = pd.DataFrame({
    "Dataset": ["CMAPSS", "FEMTO", "IMS", "EDP", "METRO", "Navarchos", "XJTU", "BHD", "AZURE", "Formula 1"],
    # "type": ["s", "e", "e", "r", "r", "r", "e", "r", "s", "s", "r"],
    "#records": [265256, 21493, 9464, 209236, 1516948, 854178, 9216, 6626869, 876100, 5563727],
    "Min scenario length": [19, 172, 984, 52244, 1516948, 1482, 42, 11, 8761, 4809],
    "Avg scenario length": [187, 1264, 3154, 52309, 1516948, 32853, 614, 1529, 8761, 22344],
    "Std scenario length": [82, 772, 2806, 50, 0, 46824, 853, 776, 0, 10595],
    "Min dimensions": [14, 44, 88, 79, 15, 6, 44, 9, 4, 17],
    "Max dimensions": [21, 44, 176, 79, 15, 6, 44, 32, 4, 17],
    "failures": [709, 6, 3, 8, 4, 21, 15, 4334, 761, 249],
    "scenarios with failure": [709, 6, 3, 4, 1, 13, 15, 4334, 100, 249],
    "scenarios without failure": [707, 11, 0, 0, 0, 13, 0, 0, 0, 0],
    "PH": [13, 52, 99, 8640, 17280, 10800, 18, 10, 96, 1920],
    # "maximum PH": [42, 169, 324, 8640, 60480, 21600, 19, 20, 192, 4800],
    "lead": [2, 2, 2, 288, 720, 720, 2, 2, 2, 960]
})

# Calculate standard deviation and top-1/top-3 labels
std_of_method_performance = df.groupby(['Dataset', 'Technique', 'Flavor'], as_index=False)['AD1_AUC'].std(ddof=0).rename(columns={'AD1_AUC': 'Std'})

df = df.merge(std_of_method_performance, on=['Dataset', 'Technique', 'Flavor'], how='left')

# top-1 performance calculation
df['is_max'] = (
    df['AD1_AUC']
      .eq(df.groupby(['Dataset','Technique','Flavor'])['AD1_AUC'].transform('max'))
      .astype(int)
)

df= df[df['is_max'] > 0]

# top-3 performance calculation
df['rank_desc'] = (
    df.groupby(['Dataset'])['AD1_AUC']
      .rank(method='dense', ascending=False)
)

df['is_top3'] = (df['rank_desc'] <= 3).astype(int)

df.drop(columns='rank_desc', inplace=True)

df = df.merge(dataset_characteristics_df, on=['Dataset'], how='left')

assert not df[dataset_characteristics_df.columns.tolist()].isna().any().any()

categorical_cols = ['Flavor', 'Technique']#, 'type']

df_encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)

df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

task = 'rf'
if task == 'classification':
    target_variable = 'is_top3'
    secondary_target_variable = 'is_max'
    X = df.drop(columns=[target_variable, secondary_target_variable, 'Dataset', 'AD1_AUC', 'Duration', 'Std'])
    y = df[target_variable]

    X_train, y_train = X, y
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    # )

    param_space = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(3, 10)),# + [None],
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 20),
        # skip min_weight_fraction_leaf
        'max_features': ['sqrt', 'log2', None],
        'random_state': [RANDOM_STATE],
        # skip max_leaf_nodes
        # skip min_impurity_decrease
        'class_weight': ['balanced'],
        # skip ccp_alpha
        # skip monotonic_cst
    }

    @scheduler.parallel(n_jobs=12)
    def objective_function(**params):
        results = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        # for params in args_list:
        clf = DecisionTreeClassifier(
            criterion=params['criterion'],
            splitter=params['splitter'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=params['random_state'],
            class_weight=params['class_weight'],
        )

        # F1 score for imbalanced classification
        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1_macro')
        results.append(np.mean(scores))

        return results


    tuner = Tuner(param_space, objective_function, conf_dict={
        'initial_random': 12,
        'num_iteration': 125,
        'batch_size': 8
        # 'initial_random': 1,
        # 'num_iteration': 1,
        # 'batch_size': 1
    })
    results = tuner.maximize()

    print("Best Parameters:", results['best_params'])
    print("Best F1 Score:", results['best_objective'])

    best_clf = DecisionTreeClassifier(
        **results['best_params']
    )

    best_clf.fit(X_train, y_train)
    # y_pred = best_clf.predict(X_test)
    #
    #
    # from sklearn.metrics import classification_report
    #
    # print(classification_report(y_test, y_pred))

    # feature importance
    feature_names = X.columns.tolist()
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(8, 4))
    plt.title("Feature Importances")
    plt.bar(range(len(sorted_importances)), sorted_importances, color='skyblue')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=90)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    from sklearn.tree import plot_tree

    plt.figure(figsize=(100, 100))
    plot_tree(
        best_clf,
        feature_names=X.columns.tolist(),
        class_names=[f'Not {"top-3" if target_variable == "is_top3" else "max"} performance', f'Top-3 {"top-3" if target_variable == "is_top3" else "max"} performance'],
        filled=True,
        impurity=True
    )
    plt.show()
elif task == 'regression':
    # Decision Tree Regressor
    target_variable = 'AD1_AUC'
    X = df.drop(columns=[target_variable, 'is_top3', 'Dataset', 'is_max'])
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    param_space = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(3, 10)),# + [None],
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 20),
        # skip min_weight_fraction_leaf
        'max_features': ['sqrt', 'log2', None],
        'random_state': [RANDOM_STATE],
        # skip max_leaf_nodes
        # skip min_impurity_decrease
        # skip ccp_alpha
        # skip monotonic_cst
    }


    @scheduler.parallel(n_jobs=12)
    def objective_function(**params):
        results = []
        skf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        # for params in args_list:
        clf = DecisionTreeRegressor(
            criterion=params['criterion'],
            splitter=params['splitter'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=params['random_state'],
        )

        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='neg_root_mean_squared_error')
        results.append(np.mean(scores))

        return results


    tuner = Tuner(param_space, objective_function, conf_dict={
        'initial_random': 12,
        'num_iteration': 125,
        'batch_size': 8
        # 'initial_random': 1,
        # 'num_iteration': 1,
        # 'batch_size': 1
    })
    results = tuner.maximize()

    print("Best Parameters:", results['best_params'])
    print("Best RMSE:", results['best_objective'])

    best_clf = DecisionTreeRegressor(
        **results['best_params']
    )

    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    from sklearn.tree import plot_tree

    plt.figure(figsize=(100, 100))
    plot_tree(best_clf, feature_names=X.columns.tolist(), filled=True, impurity=True)
    plt.show()
elif task == 'figs':
    from imodels import FIGSClassifier
    from sklearn.tree import plot_tree

    target_variable = 'is_top3'
    secondary_target_variable = 'is_max'
    X = df.drop(columns=[target_variable, secondary_target_variable, 'Dataset', 'AD1_AUC', 'Duration', 'Std'])
    y = df[target_variable]

    X_train = X_test = X
    y_train = y_test = y
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=RANDOM_STATE
    # )

    param_space = {
        'max_rules': range(1, 20),
        'max_trees': range(1, 100),
        'max_features': ['sqrt', 'log2', None],
        'random_state': [RANDOM_STATE],
    }


    @scheduler.parallel(n_jobs=12)
    def objective_function(**params):
        results = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        # for params in args_list:
        clf = FIGSClassifier(
            max_rules=params['max_rules'],
            max_trees=params['max_trees'],
            max_features=params['max_features'],
            random_state=params['random_state'],
        )

        # F1 score for imbalanced classification
        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1_macro')
        results.append(np.mean(scores))

        return results


    tuner = Tuner(param_space, objective_function, conf_dict={
        'initial_random': 12,
        'num_iteration': 125,
        'batch_size': 8
        # 'initial_random': 1,
        # 'num_iteration': 1,
        # 'batch_size': 1
    })
    results = tuner.maximize()

    print("Best Parameters:", results['best_params'])
    print("Best F1 Score:", results['best_objective'])

    best_clf = FIGSClassifier(
        **results['best_params']
    )

    best_clf.fit(X_train, y_train)

    print(best_clf)  # print the model

    best_clf.plot(feature_names=X.columns.tolist(), filename='figs_out.svg', dpi=300)

    y_pred = best_clf.predict(X_test)

    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))

    from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
    import dtreeviz

    tree_list = []
    for i in range(len(best_clf.trees_)):
        current_tree = extract_sklearn_tree_from_figs(best_clf, tree_num=i, n_classes=2)
        tree_list.append(current_tree)

        viz_model = dtreeviz.model(
            current_tree,
            X_train=X_train,
            y_train=y_train,
            feature_names=X_train.columns.tolist(),
            target_name='is_top3',
            class_names=['Not top-3 performance', 'Top-3 performance'],
        )

        v = viz_model.view()
        v.show()
        v.save(f"no_data_just_pdm_case_{i}.svg")

    best_clf = tree_list
elif task == 'rf':
    from imodels import FIGSClassifier
    from imodels.importance import RandomForestPlusClassifier
    from sklearn.tree import plot_tree

    target_variable = 'is_top3'
    secondary_target_variable = 'is_max'
    X = df.drop(columns=[target_variable, secondary_target_variable, 'Dataset', 'AD1_AUC'])#, 'Duration', 'Std'])
    y = df[target_variable]

    X_train = X_test = X
    y_train = y_test = y
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=RANDOM_STATE
    # )

    param_space = {
        'n_estimators': range(1, 100),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': list(range(3, 10)),  # + [None],
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 20),
        # skip min_weight_fraction_leaf
        'max_features': ['sqrt', 'log2', None],
        # skip max_leaf_nodes
        # skip min_impurity_decrease
        # skip bootstrap
        # skip oob_score
        # skip n_jobs
        'random_state': [RANDOM_STATE],
        # skip verbose
        # skip warm start
        'class_weight': ['balanced'],
        # skip ccp_alpha
        # skip max_samples
        # skip monotonic_cst
    }

    @scheduler.parallel(n_jobs=12)
    def objective_function(**params):
        results = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        # for params in args_list:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=params['random_state'],
            class_weight=params['class_weight'],
        )

        # F1 score for imbalanced classification
        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1_macro')
        results.append(np.mean(scores))

        return results


    tuner = Tuner(param_space, objective_function, conf_dict={
        'initial_random': 12,
        'num_iteration': 60,
        'batch_size': 12
        # 'initial_random': 1,
        # 'num_iteration': 1,
        # 'batch_size': 1
    })
    results = tuner.maximize()

    print("Best Parameters:", results['best_params'])
    print("Best F1 Score:", results['best_objective'])

    best_clf = RandomForestClassifier(
        **results['best_params']
    )

    X_train_df = X_train
    X_train = X_train.astype(int)

    rf_plus_model = RandomForestPlusClassifier(rf_model=best_clf)
    rf_plus_model.fit(X_train.to_numpy(), y_train.to_numpy())

    y_pred = rf_plus_model.predict(X_test.astype(int).to_numpy())

    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))

    mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X_train.to_numpy(), y_train.to_numpy())
    print(X_train_df.columns.tolist())
    print(mdi_plus_scores.sort_values("importance", ascending=False))
    exit(-1)


from sklearn.tree import _tree

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


if type(best_clf) is list:
    for index, clf in enumerate(best_clf):
        rules = get_rules(clf, X.columns.tolist(), ['Not top-3 performance', 'Top-3 performance'] if not task == 'regression' else None)
        print(f'index: {index}')
        for r in rules:

            print(r)
else:
    rules = get_rules(best_clf, X.columns.tolist(), ['Not top-3 performance', 'Top-3 performance'] if not task == 'regression' else None)
    for r in rules:
        print(r)