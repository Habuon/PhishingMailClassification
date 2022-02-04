from models.AbstractModel import AbstractModel


class BaseLine(AbstractModel):
    def __init__(self):
        self.best = None

    def fit(self, train_x, train_y, finetune=False):
        mx = 0
        best = None
        for value in set(train_y):
            count = list(train_y).count(value)
            if count > mx:
                mx = count
                best = value
        self.best = best

    def evaluate(self, test_x, test_y):
        mx = 0
        best = None
        for value in set(test_y):
            count = list(test_y).count(value)
            if count > mx:
                mx = count
                best = value
        self.best = best
        return list(test_y).count(best) / len(test_y) * 100

    def predict(self, x):
        return self.best

    def __str__(self):
        return " BaseLine "
