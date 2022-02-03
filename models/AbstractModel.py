from abc import ABC, abstractmethod


class AbstractModel(ABC):
    BANNER_WIDTH = 50

    @abstractmethod
    def fit(self, train_x, train_y):
        pass

    @abstractmethod
    def evaluate(self, test_x, test_y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def get_banner(self, msg):
        s = str(self)
        p, t = (self.BANNER_WIDTH - len(s)) // 2, (self.BANNER_WIDTH - len(s)) % 2
        if t != 0:
            result = "-" * p + s + " " + "-" * p + "\n"
        else:
            result = "-" * p + s + "-" * p + "\n"
        result += "| " + msg + " " * (self.BANNER_WIDTH - len(msg) - 3) + "|\n"
        result += "-" * self.BANNER_WIDTH + "\n"
        return result





