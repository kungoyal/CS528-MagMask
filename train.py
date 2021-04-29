from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import  GaussianNB

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import Normalizer

SELECTION_METRIC = "F1"

def load_data():
    pass

    X = None    # Feature vector array
    y = None    # Multi-class labels

    return X, y

def get_scores(y_true, y_pred):
    scores_arr = []
    scores_arr.append(["Accuracy", accuracy_score(y_true, y_pred)])
    scores_arr.append(["Precision", precision_score(y_true, y_pred)])
    scores_arr.append(["Recall", recall_score(y_true, y_pred)])
    scores_arr.append(["F1", f1_score(y_true, y_pred, average='weighted')])
    scores_arr.append(["ROC_AUC", roc_auc_score(y_true, y_pred)])

    return scores_arr

X, y = load_data()
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier_list = [
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, max_depth=4)),
    ("LinearSVC", LinearSVC()),
    ("RBF SVC", SVC()),
    ("GaussianNB", GaussianNB()),

]


# not ideal because data has class imbalance
kf = KFold()
fold = 0
fold_scores = {}
for train_idx, test_idx in kf.split(X_tr):
    fold += 1
    fold_scores[fold] = {}
    X_train, y_train = X_tr[train_idx], y_tr[train_idx]
    X_valid, y_valid = X_tr[train_idx], y_tr[train_idx]

    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_valid = normalizer.transform(X_test)

    print(f"Fold = {fold}")
    for name, model in classifier_list:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        scores = get_scores(y_valid, y_pred)
        fold_scores[fold][name] = scores
        print(f"\tClassifier: {name}")
        for n, s in scores:
            print(f"\t\t{n}\t:{s}")

# calculate average over all folds

# identify best scoring model

# save best scoring model







