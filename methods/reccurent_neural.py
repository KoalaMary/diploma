import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb


class ReccurentNeuralMethod:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_rc(self):
        # (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=100)
        # data = np.concatenate((training_data, testing_data), axis=0)
        # targets = np.concatenate((training_targets, testing_targets), axis=0)
        #
        # def vectorize(sequences, dimension=100):
        #     results = np.zeros((len(sequences), dimension))
        #     for i, sequence in enumerate(sequences):
        #         results[i, sequence] = 1
        #     return results
        #
        # data = vectorize(data)
        # targets = np.array(targets).astype("float32")
        # test_x = data[:160]
        # test_y = targets[:160]
        # train_x = data[47000:]
        # train_y = targets[47000:]

        model = models.Sequential()
        # Input - Layer
        model.add(layers.Dense(50, activation="relu", input_shape=(600,)))
        # Hidden - Layers
        model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
        model.add(layers.Dense(50, activation="relu"))
        model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
        model.add(layers.Dense(50, activation="relu"))
        # Output- Layer
        model.add(layers.Dense(1, activation="sigmoid"))
        model.summary()

        # compiling the model
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        #######################################

        # model.add(layers.Dense(50, activation="relu", input_shape=(40725,)))
        # model.add(
        #     layers.Embedding(input_dim=10000,
        #                      input_length=10000,
        #                      output_dim=100,
        #                      # weights=[embedding_matrix],
        #                      trainable=False,
        #                      mask_zero=True))
        #
        # # Masking layer for pre-trained embeddings
        # model.add(layers.Masking(mask_value=0.0))
        #
        # # Recurrent layer
        # model.add(layers.LSTM(100, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
        #
        # # Fully connected layer
        # model.add(layers.Dense(100, activation='relu'))
        #
        # # Dropout for regularization
        # model.add(layers.Dropout(0.5))
        #
        # # Output layer
        # model.add(layers.Dense(100, activation='softmax'))
        #
        # # Compile the model
        # model.compile(
        #     optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        ############################################
        #
        # results = model.fit(
        #     train_x, train_y,
        #     epochs=2,
        #     batch_size=32,
        #     validation_data=(test_x, test_y)
        # )

        results = model.fit(
            self.X_train, self.y_train,
            epochs=2,
            batch_size=32,
            validation_data=(self.X_test, self.y_test)
        )
        print("Test-Accuracy:", np.mean(results.history["val_acc"]))
