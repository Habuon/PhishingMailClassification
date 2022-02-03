from sklearn import svm
from sklearn.model_selection import train_test_split

from models.AbstractModel import AbstractModel


class SVC(AbstractModel):
    KERNELS = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    MAX_C = 25
    MAX_GAMMA = 25
    STEP_C = 1
    STEP_G = 1

    def __init__(self):
        self.c_g_k = [2, 1, 'rbf']
        C, gamma, kernel = self.c_g_k
        self.svc = svm.SVC(C=C, kernel=kernel, gamma=gamma)
        self.trained = False

    def fit(self, train_x, train_y, finetune=False):
        self.trained = True
        if finetune:
            self._finetune(train_x, train_y)
        else:
            self.svc.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
        assert self.trained, "Model has not been trained yet"
        return self.svc.score(test_x, test_y) * 100

    def predict(self, x):
        assert self.trained, "Model has not been trained yet"
        return self.svc.predict(x)

    def _finetune(self, train_x, train_y):
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)
        best_score = 0
        c_g_k = []
        for kernel in self.KERNELS:
            c = 1
            while c < self.MAX_C:
                gamma = 1
                while gamma < self.MAX_GAMMA:
                    svc = svm.SVC(C=c, kernel=kernel, gamma=gamma)
                    svc.fit(train_x, train_y)
                    score = svc.score(test_x, test_y)
                    if score > best_score:
                        best_score = score
                        c_g_k = [c, gamma, kernel]
                        self.svc = svc
                        self.c_g_k = c_g_k
                    gamma += self.STEP_G
                c += self.STEP_C

        print(f"for svc best c : {c_g_k[0]}, best gamma: {c_g_k[1]} and best kernel: {c_g_k[2]}")

    def __str__(self):
        return " Support Vector Classifier "
