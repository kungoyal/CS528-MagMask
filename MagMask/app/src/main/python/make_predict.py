import extract_features
def get_pred(model, input_csv):
    classifier, normalizer = model
    input_fv = extract_features.get_fv_csv(input_csv)
    pred_arr = []
    for fv in input_fv:
        norm_inp = normalizer.transform(fv)
        pred = classifier.predict(norm_inp)
        pred_arr.append(pred)
    return pred_arr