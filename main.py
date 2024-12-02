from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import mean_squared_error
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


model = Sequential(
    [
        Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(10, activation="relu"),
        Dense(15, activation="relu"),
        Dense(10, activation="linear"),
    ]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100)
prediction = model.predict(x_test)
predicted_classes = np.argmax(prediction, axis=1)

accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy: {accuracy * 100}%")

mse = mean_squared_error(y_test, predicted_classes)
print(f"Mean Square Error: {mse}")
