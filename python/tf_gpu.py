import tensorflow as tf

# GPU版Tensor Flowを、特定のGPUで実行する

GPU_INDEX = 2

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        print(gpus)
        print(logical_gpus)
    except RuntimeError as e:
        print(e)

try:
    with tf.device('/device:GPU:{}'.format(GPU_INDEX)):  # GPUの番号を指定する

        # MNIST
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test)
except RuntimeError as e:
    print(e)
