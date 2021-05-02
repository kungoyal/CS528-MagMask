import pickle, os
from os.path import dirname, join

classifier_path= join(dirname(__file__), "best_classifier.pkl")
normalizer_path = join(dirname(__file__), "best_classifier.pkl")
# classifier_path = __file__+'/../best_classifier.pkl'
# normalizer_path = __file__+'/../normalizer.pkl'

def get_classifier_normalizer():
    with open (normalizer_path, 'rb') as fil:
        normalizer = pickle.load(fil)

    with open(classifier_path, 'rb') as fil:
        classifier = pickle.load(fil)

    return classifier, normalizer

if __name__ == "__main__":
    c, n = get_classifier_normalizer()
    print(c)
    print(n)
    print(c.predict)
    print(n.transform)