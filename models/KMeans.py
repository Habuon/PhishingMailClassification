from sklearn import cluster
from sklearn.model_selection import train_test_split

from models.AbstractModel import AbstractModel


class KMeans(AbstractModel):
    MAX_N = 100
    STEP_N = 1
    ATTEMPTS = 10

    def __init__(self):
        self.n_clusters = 88
        self.kmeans = None
        self.labels = None
        self.trained = False

    def fit(self, train_x, train_y, finetune=False):
        self.trained = True
        if finetune:
            self._finetune(train_x, train_y)
        else:
            self.kmeans = cluster.KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0)
            self.kmeans.fit(train_x)
            self.labels = KMeans._label_centers(self.kmeans.cluster_centers_, self.kmeans.labels_, train_y)

    def evaluate(self, test_x, test_y):
        assert self.trained, "Model has not been trained yet"
        return KMeans._score(self.kmeans, self.labels, test_x, test_y)

    def predict(self, x):
        assert self.trained, "Model has not been trained yet"
        return self.labels[self.kmeans.predict(x)]

    def _finetune(self, train_x, train_y):
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)
        best = 0

        n = 1
        while n < self.MAX_N:
            kmeans = cluster.KMeans(n_clusters=n, init='k-means++', random_state=1)
            kmeans.fit(train_x)
            labels = KMeans._label_centers(kmeans.cluster_centers_, kmeans.labels_, train_y)
            score = KMeans._score(kmeans, labels, test_x, test_y)
            if score > best:
                self.kmeans = kmeans
                self.labels = labels
                best = score
                self.n_clusters = n
                print(n, score)

            n += self.STEP_N

    @staticmethod
    def _label_centers(centers, labels, y):
        result = []
        for center in range(len(centers)):
            positive = 0
            negative = 0
            for i in range(len(labels)):
                if labels[i] != center:
                    continue
                if y[i] == 0:
                    negative += 1
                else:
                    positive += 1
            result.append(0 if negative > positive else 1)
        return result

    @staticmethod
    def _score(kmeans, labels, test_x, test_y):
        good = 0
        predictions = kmeans.predict(test_x)
        for i, prediction in enumerate(predictions):
            if test_y[i] == labels[prediction]:
                good += 1
        return good / len(test_x) * 100

    def __str__(self):
        return " K-means Clustering "
