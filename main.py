from sklearn.model_selection import train_test_split

import numpy as np

from models.BaseLine import BaseLine
from models.KMeans import KMeans
from models.MLP import MLP
from models.RFC import RFC
from models.SVC import SVC

from preprocessing import get_data


def main():

    x = get_data()
    np.random.shuffle(x)
    y = x[:, -1]
    x = np.delete(x, -1, 1)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=1)

    models = [BaseLine(), SVC(), RFC(), KMeans(), MLP(train_x.shape[1])]
    finetune = [False, False, False, False, False]

    for i in range(len(models)):
        model = models[i]
        model.fit(train_x, train_y, finetune[i])
        msg = f"Test accuracy: {model.evaluate(test_x, test_y)}"
        print(model.get_banner(msg))


main()

