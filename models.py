from tensorflow import keras

def build_mlp(input_shape=(25,25,1)):
    """
    A simple feed-forward network:
    - Flattens the 25×25 grid
    - Two ReLU-activated
    - One linear output (distance or count)
    """
    return keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)   # regression/count output
    ])

def build_cnn(input_shape=(25,25,1)):
    """
    A small ConvNet:
    - Conv → ReLU → MaxPool
    - Conv → ReLU → MaxPool
    - Flatten → Dense → Output
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)
