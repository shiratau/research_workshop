from lib import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(df):
    x_train, y_train, x_test, y_test = _split_sets_for_run(df)
    model = _build_model()
    _train(model, x_train, y_train)
    predictions = _predict(model, x_test)
    return predictions, y_test


def _split_sets_for_run(df):
    train_set = df.sample(frac=0.8)
    print("Train set:", train_set.shape)
    test_set = df.drop(train_set.index)
    print("Test set:", test_set.shape)

    x_train = tf.ragged.constant([v[..., None] for v in train_set["input"].values])
    x_test = tf.ragged.constant([v[..., None] for v in test_set["input"].values])
    print(x_train.bounding_shape())
    print(x_train[0].shape)
    print(x_train[1].shape)
    # print(x_train[2].shape)

    y_train = np.asarray(train_set["output"]).astype(np.float32)
    y_test = np.asarray(test_set["output"]).astype(np.float32)
    print(y_train)
    print(y_train.shape)

    return x_train, y_train, x_test, y_test


def _build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(None, 1)))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=32))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    return model


def _train(model, x_train, y_train):
    history = model.fit(x=x_train, y=y_train, epochs=30)

    his = history.history
    print(his.keys())
    plt.plot(his["loss"], label="loss")
    plt.legend(loc="upper right")
    plt.savefig('loss_single.png', bbox_inches='tight')


def _predict(model, x_test):
    prediction = model.predict(x_test)
    # print(prediction)
    return prediction
