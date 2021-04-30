from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import Normalizer

import pandas as pd
import pickle

# Classifier with best selection metric will be saved into "best_classifier.pkl"
SELECTION_METRIC = "F1"


def load_data():
    # TODO: open data files, perform feature extraction and store feature vectors and labels in X, y

    X = None  # Feature vector array
    y = None  # Multi-class labels

    return X, y


def get_scores(y_true, y_pred):
    sd = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average='weighted'),
        "ROC_AUC": roc_auc_score(y_true, y_pred)
    }  # scores dict

    return sd


X, y = load_data()
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


