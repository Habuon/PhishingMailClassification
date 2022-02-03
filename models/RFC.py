from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from models.AbstractModel import AbstractModel


class RFC(AbstractModel):
    MAX_E = 50
    MAX_D = 50
    STEP_E = 1
    STEP_D = 1

    def __init__(self):
        self.e_d = [26, 12]
        e, d = self.e_d
        self.rfc = RandomForestClassifier(n_estimators=e, max_depth=d)
        self.trained = False

    def evaluate(self, test_x, test_y):
        assert self.trained, "Model has not been trained yet"
        return self.rfc.score(test_x, test_y) * 100

    def predict(self, x):
        assert self.trained, "Model has not been trained yet"
        return self.rfc.apply(x)

    def fit(self, train_x, train_y, finetune=False):
        self.trained = True
        if finetune:
            self._finetune(train_x, train_y)
        else:
            self.rfc.fit(train_x, train_y)

    def _finetune(self, train_x, train_y):
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)
        best_score = 0
        e_d = []
        e = 1
        while e < self.MAX_E:
            d = 1
            while d < self.MAX_D:
                rfc = RandomForestClassifier(n_estimators=e, max_depth=d)
                rfc.fit(train_x, train_y)
                score = rfc.score(test_x, test_y)
                if score > best_score:
                    best_score = score
                    e_d = [e, d]
                    self.rfc = rfc
                    self.e_d = e_d
                    print(e_d, score)
                d += self.STEP_D
            e += self.STEP_E
        print(f"for rfc best e : {e_d[0]} and best d: {e_d[1]}")

    def __str__(self):
        return " Random Forest Classifier "
