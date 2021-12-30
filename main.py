from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from matplotlib import pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

import numpy as np

from sklearn import svm


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


def get_svc(X, y):
    c_g_k = [2, 1, 'rbf']
    C, gamma, kernel = c_g_k
    svc = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    svc.fit(X, y)

    return svc


def get_rfc(X, y):
    e_d = [18, 12]
    e, d = e_d
    rfc = RandomForestClassifier(n_estimators=e, max_depth=d)
    rfc.fit(X, y)
    return rfc


def get_mlp(X, y):
    mlp = Sequential()
    mlp.add(Dense(16, activation='sigmoid', input_dim=X.shape[1]))
    mlp.add(Dense(64, activation='sigmoid'))
    mlp.add(Dense(2, activation='softmax'))

    mlp.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

    history = mlp.fit(X, tf.keras.utils.to_categorical(y), epochs=50, validation_split=0.1, verbose=False)

    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()
    return mlp



def main():

    X = get_data()
    np.random.shuffle(X)
    y = X[:, -1]
    X = np.delete(X, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    print("baseline: ", score(y_test, baseline(X_test, y_test)))

    print(("-" * 20) + "SVC" + ("-" * 20))
    svc = get_svc(X_test, y_test)
    print("Test accuracy:", svc.score(X_test, y_test))
    print(("-" * 20) + "---" + ("-" * 20))
    print(("-" * 20) + "RFC" + ("-" * 20))
    rfc = get_rfc(X_train, y_train)
    print("Test accuracy:", rfc.score(X_test, y_test))
    print(("-" * 20) + "---" + ("-" * 20))
    print(("-" * 20) + "MLP" + ("-" * 20))
    mlp = get_mlp(X_train, y_train)
    print("Test accuracy:", mlp.evaluate(X_test, tf.keras.utils.to_categorical(y_test))[1])
    print(("-" * 20) + "---" + ("-" * 20))


main()
