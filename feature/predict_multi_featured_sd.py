from lib import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(df):
    train_dataset, train_y, test_dataset, test_y = _split_sets_for_run(df)
    model = _build_model()
    _train(model, train_dataset, test_dataset)
    predictions = _predict(model, test_dataset)
    return predictions, test_y


def _split_sets_for_run(df):
    train_set = df.sample(frac=0.8, random_state=42)
    test_set = df.drop(train_set.index)

    # Prepare the datasets
    train_features_samples = tf.ragged.constant(train_set["input_samples"].values.tolist(), dtype=tf.float64)
    train_features_timestamps = tf.ragged.constant(train_set["input_timestamps"].values.tolist(), dtype=tf.float64)
    train_y = tf.ragged.constant(train_set["output"].values.tolist(), dtype=tf.float64)

    test_features_samples = tf.ragged.constant(test_set["input_samples"].values.tolist(), dtype=tf.float64)
    test_features_timestamps = tf.ragged.constant(test_set["input_timestamps"].values.tolist(), dtype=tf.float64)
    test_y = tf.ragged.constant(test_set["output"].values.tolist(), dtype=tf.float64)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features_samples, train_features_timestamps, train_y)).batch(2)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features_samples, test_features_timestamps, test_y)).batch(2)

    train_dataset = train_dataset.map(_process_element)
    test_dataset = test_dataset.map(_process_element)

    test_y = np.array([v for v in test_set["output"].values])

    return train_dataset, train_y, test_dataset, test_y


def _build_model():
    model = CombinedLSTMModel()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mae')
    model.build(input_shape=[(None, None, 1), (None, None, 1)])  # Specify the input shape
    model.summary()

    # _shape_visualization(model) # doesn't work here

    return model


def _train(model, train_dataset, test_dataset):
    history = model.fit(train_dataset, epochs=20, validation_data=test_dataset)

    his = history.history
    print(his.keys())
    plt.plot(his["loss"], label="loss")
    plt.legend(loc="upper right")
    plt.savefig('loss.png', bbox_inches='tight')


def _predict(model, test_dataset):
    prediction = model.predict(test_dataset)
    return prediction


def _shape_visualization(model):
    tf.keras.utils.plot_model(model, to_file='model_shape.png', show_shapes=True)


class CombinedLSTMModel(tf.keras.Model):
    def __init__(self):
        super(CombinedLSTMModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        # Adding binning layer didn't give better results. You can try it by uncomment the line below.
        # self.discretization = tf.keras.layers.Discretization(num_bins=10, epsilon=0.01)
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Discretization
        self.dense2 = tf.keras.layers.Dense(3, activation='linear')

    def call(self, inputs):
        samples, timestamps = inputs
        x = tf.concat([samples, timestamps], axis=-1)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        return self.dense2(x)


def _process_element(samples, timestamps, labels):
    # Expand dimensions to match LSTM input requirements
    samples = tf.expand_dims(samples, axis=-1)
    timestamps = tf.expand_dims(timestamps, axis=-1)
    return (samples, timestamps), labels

