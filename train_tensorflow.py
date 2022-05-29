import neptune.new as neptune
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# initialize netptune
run = neptune.init(
    project="327485253/starter",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmODEyNTkxZi1jMWQ5LTQ2NmMtOTE4ZC0yZmQxMTU5NzYyYjYifQ==",
    source_files=['train_tensorflow.py', 'requirements.txt']
)

# training: load data
mnist = tf.keras.datasets.mnist

# training: split data into train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# training: build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# training: compile model
model.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# netptune: log epoch and batch

class NeptuneLogger(Callback):
    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            run["batch/{}".format(log_name)].log(log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            run["epoch/{}".format(log_name)].log(log_value)



EPOCH_NR = 5
BATCH_SIZE = 32

# run
run["parameters/epoch_nr"] = EPOCH_NR
run["parameters/batch_size"] = BATCH_SIZE
run["sys/name"] = "keras-metrics"
run["sys/tags"].add("advanced")


history = model.fit(
    x=x_train,
    y=y_train,
    epochs=EPOCH_NR,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=[NeptuneLogger()],
)


y_test_pred = np.asarray(model.predict(x_test))
y_test_pred_class = np.argmax(y_test_pred, axis=1)

# print => run
run["test/f1"] = f1_score(y_test, y_test_pred_class, average="micro")

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_test, y_test_pred_class, ax=ax)

# plt.show() => run
run["diagnostic_charts"].log(neptune.types.File.as_image(fig))

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y_test, y_test_pred, ax=ax)

# plt.show() => run
run["diagnostic_charts"].log(neptune.types.File.as_image(fig))

# 同步一下
run.stop()