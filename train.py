from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import Normalizer

import pandas as pd
import numpy as np
import pickle
import os

import extract_features
# Classifier with best selection metric will be saved into "best_classifier.pkl"
SELECTION_METRIC = "Accuracy"


def load_data(data):
    # TODO: open data files, perform feature extraction and store feature vectors and labels in X, y
    columns = ['Initials', 'Environment ID', 'Action ID', 'Attempt', 'Window', 'Accelerometer', 'Gyroscope', 'Magnetometer']

    table = []
    for f in os.listdir(data):
        print(os.path.join(data, f))
        meta = os.path.splitext(f)[0].split("_")
        d = pd.read_csv(os.path.join(data, f), header=None)
        d.columns = ['x', 'y', 'z', 'time']
        split_idx = list(np.where(d.isnull().any(axis=1))[0])
        sensor1_fv = extract_features.get_feature_vectors(d.iloc[0:split_idx[0]].values.astype(float))
        sensor2_fv = extract_features.get_feature_vectors(d.iloc[split_idx[0]+1:split_idx[1]].values.astype(float))
        sensor3_fv = extract_features.get_feature_vectors(d.iloc[split_idx[1]+1: split_idx[2]].values.astype(float))
        rows = []
        for win in range(len(sensor1_fv)):
            rows.append(meta + [win, sensor1_fv[win].tolist(), sensor2_fv[win].tolist(), sensor3_fv[win].tolist()])
        table.extend(rows)
    data_df = pd.DataFrame(table, columns=columns)
    data_cols = ['Accelerometer', 'Gyroscope', 'Magnetometer']
    label_cols = ['Action ID']
    X = np.array(data_df[data_cols].values.tolist()).reshape(len(table), -1)  # Feature vector array
    y = data_df[label_cols].values.reshape(-1) # Multi-class labels
    # input()
    return X, y


def get_scores(y_true, y_pred):
    # try:
    sd = {
        "Accuracy": accuracy_score(y_true, y_pred),
        # "Precision": precision_score(y_true, y_pred),
        # "Recall": recall_score(y_true, y_pred),
        # "F1": f1_score(y_true, y_pred, average='weighted'),
        # "ROC_AUC": roc_auc_score(y_true, y_pred)
    }  # scores dict

    return sd

def main(data_dir):
    X, y = load_data(data_dir)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add multiple classifiers here and the classifier with best hyperparameters will be selected automatically
    classifier_dict = {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=4),
        "LinearSVC": LinearSVC(),
        "RBF SVC": SVC(),
        "GaussianNB": GaussianNB(),
    }

    # TODO: up-sample minority class or use other methods to deal with class imbalance
    kf = KFold()  # not ideal because data has class imbalance
    fold = 0
    fold_scores = {}
    for train_idx, test_idx in kf.split(X_tr):
        fold += 1
        fold_scores[fold] = {}
        X_train, y_train = X_tr[train_idx], y_tr[train_idx]
        X_valid, y_valid = X_tr[train_idx], y_tr[train_idx]

        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train)
        X_valid = normalizer.transform(X_valid)

        print(f"Fold = {fold}")
        for name in classifier_dict.keys():
            model = classifier_dict[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            scores = get_scores(y_valid, y_pred)
            fold_scores[fold][name] = scores

            print(f"\tClassifier: {name}")
            for metric in scores.keys():
                print(f"\t\t{metric}\t:{scores[metric]}")

    # calculate average over all folds
    scores_df = pd.DataFrame.from_dict(fold_scores).T
    print(f"Total number of folds = {fold}")

    best_classifier = ""
    best_f1 = 0
    for classifier in scores_df.keys():
        agg_metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0, "ROC_AUC": 0}
        for fol in scores_df[classifier]:
            for metr in fol.keys():
                agg_metrics[metr] += (fol[metr]/fold)

        print(f"Classifier = {classifier}")
        for agg_m in agg_metrics.keys():
            print(f"\t{agg_m} = {agg_metrics[agg_m]}")

            # identify best scoring model
            if agg_m == SELECTION_METRIC:
                if agg_metrics[agg_m] > best_f1:
                    best_f1 = agg_metrics[agg_m]
                    best_classifier = classifier

    # re-train and save best scoring model

    normalizer = Normalizer()
    X_tr = normalizer.fit_transform(X_tr)
    X_test = normalizer.transform(X_test)

    best_model = classifier_dict[best_classifier]
    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_test)
    scores = get_scores(y_test, y_pred)
    print(f"Best Classifier: {best_classifier}")
    for metric in scores.keys():
        print(f"\t\t{metric}\t:{scores[metric]}")

    with open("best_classifier.pkl", "wb") as fil:
        pickle.dump(best_model, fil)

    # Required since data needs to be normalized before prediction
    with open("normalizer.pkl", "wb") as fil:
        pickle.dump(normalizer, fil)

    print("DONE")


if __name__ == "__main__":
    data_dir = "./Data"
    load_data(data_dir)
    print("DONE")
    main(data_dir)

