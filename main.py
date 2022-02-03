from sklearn.model_selection import train_test_split

import numpy as np

from models.KMeans import KMeans
from models.MLP import MLP
from models.RFC import RFC
from models.SVC import SVC

from preprocessing import get_data


def baseline(X, y):
    counts = dict()
    for e in y:
        if e not in counts:
            counts[e] = 0
        counts[e] += 1
    return sorted(counts.keys(), key=lambda x: counts[x], reverse=True)[0]


def score(y, value):
    counts = dict()
    for e in y:
        if e not in counts:
            counts[e] = 0
        counts[e] += 1
    return counts[value] / len(y)


def main():

    X = get_data()
    np.random.shuffle(X)
    y = X[:, -1]
    X = np.delete(X, -1, 1)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=1)

    print("baseline: ", score(test_y, baseline(test_x, test_y)))
    print()

    svc = SVC()
    svc.fit(train_x, train_y)
    msg = f"Test accuracy: {svc.evaluate(test_x, test_y)}"
    print(svc.get_banner(msg))

    rfc = RFC()
    rfc.fit(train_x, train_y)
    msg = f"Test accuracy: {rfc.evaluate(test_x, test_y)}"
    print(rfc.get_banner(msg))

    mlp = MLP(train_x.shape[1])
    mlp.fit(train_x, train_y)
    msg = f"Test accuracy: {mlp.evaluate(test_x, test_y)}"
    print(mlp.get_banner(msg))

    kmeans = KMeans()
    kmeans.fit(train_x, train_y)
    msg = f"Test accuracy: {kmeans.evaluate(test_x, test_y)}"
    print(kmeans.get_banner(msg))


main()
