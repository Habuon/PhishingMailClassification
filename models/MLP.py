import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential

from models.AbstractModel import AbstractModel


class MLP(AbstractModel):
    def __init__(self, input_shape):
        self.mlp = Sequential()
        self.mlp.add(Dense(16, activation='sigmoid', input_dim=input_shape))
        self.mlp.add(Dense(64, activation='sigmoid'))
        self.mlp.add(Dense(2, activation='softmax'))

        self.mlp.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                         metrics=['accuracy'])
        self.history = None
        self.trained = False

    def fit(self, train_x, train_y, finetune=False):
        self.trained = True
        self.history = self.mlp.fit(train_x, tf.keras.utils.to_categorical(train_y), epochs=50,
                                    validation_split=0.1, verbose=False)

    def evaluate(self, test_x, test_y):
        assert self.trained, "Model has not been trained yet"
        return self.mlp.evaluate(test_x, tf.keras.utils.to_categorical(test_y), verbose=False)[1] * 100

    def predict(self, x):
        assert self.trained, "Model has not been trained yet"
        return self.mlp.predict(x)

    def __str__(self):
        return " Multi-layer Perceptron "
