from lib import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run():
    df = pd.DataFrame(
        {
            "input": pd.Series([
                np.array([3.461, 3.478, 3.478, 3.485, 3.489, 3.489, 3.492]),
                np.array([3.469, 3.481, 3.481, 3.495, 3.495]),
                np.array([3.519, 3.528, 3.542, 3.542, 3.546, 3.55]),
                np.array([3.523, 3.529, 3.543, 3.543, 3.552, 3.554, 3.555])
            ]),
            "output": [3.281, 3.271, 3.247, 3.214]
        }
    )
    df.info()
    df.head()

    train_set = df.sample(frac=0.8)
    print("Train set:", train_set.shape)
    test_set = df.drop(train_set.index)
    print("Test set:", test_set.shape)

    x_train = tf.ragged.constant([v[..., None] for v in train_set["input"].values])
    test = tf.ragged.constant([v[..., None] for v in test_set["input"].values])
    print(x_train.bounding_shape())
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(x_train[2].shape)

    y_train = np.asarray(train_set["output"]).astype(np.float32)
    print(y_train)
    print(y_train.shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(None, 1)))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=32))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    history = model.fit(x=x_train, y=y_train, epochs=30)

    his = history.history
    print(his.keys())
    plt.plot(his["loss"], label="loss")
    plt.legend(loc="upper right")

    prediction = model.predict(test)
    print(prediction)

    print(test)


if __name__ == '__main__':
    run()
    print("done.")
