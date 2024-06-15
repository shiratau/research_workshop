from lib import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(df):
    x_train, y_train, x_test, y_test = _split_sets_for_run(df)
    # model = _build_model()
    model = _build_model_v3()
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

    y_train = tf.ragged.constant([v[..., None] for v in train_set["output"].values])
    y_test = np.array([v for v in test_set["output"].values])

    print(y_train)
    print(y_train.shape)
    print(y_test)

    return x_train, y_train, x_test, y_test


def _build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(None, 1)))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=32))
    model.add(tf.keras.layers.Dense(units=3))
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    # _shape_visualization(model)

    return model


def _build_model_v2():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 1)),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # Output layer for predicting 3 sigma values
    ])
    model.compile(loss="mse", optimizer="adam")
    # model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    # _shape_visualization(model)

    return model


def _build_model_v3():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(None, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 1)),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # Output layer for predicting 3 sigma values
    ])
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)

    # Compile the model with Adam optimizer and the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss="mse", optimizer=optimizer)
    # model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    # _shape_visualization(model)

    return model


def _train(model, x_train, y_train):
    history = model.fit(x=x_train, y=y_train, epochs=20)

    his = history.history
    print(his.keys())
    plt.plot(his["loss"], label="loss")
    plt.legend(loc="upper right")
    plt.savefig('loss.png', bbox_inches='tight')


def _predict(model, x_test):
    prediction = model.predict(x_test)
    return prediction


def _shape_visualization(model):
    tf.keras.utils.plot_model(model, to_file='model_shape.png', show_shapes=True)
