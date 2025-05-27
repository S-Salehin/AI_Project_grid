# hyperparam_tuner.py
# A simple grid search for hyperparameters over MLP and CNN models across tasks A–E

import json
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from train import load_data  # reuse data loader from your project


def tune_hyperparameters(task_name, model_type, grid):

    # Load data: only need training split
    (X_full, y_full), _ = load_data(task_name)
    # Reserve 10% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=42
    )

    best_loss = float('inf')
    best_params = {}

    # Iterating over all combinations in the grid
    for learning_rate, batch_size, hidden_size in product(
        grid['learning_rate'],
        grid['batch_size'],
        grid['units']
    ):
        # Building model based on type
        if model_type == 'MLP':
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=X_train.shape[1:]),
                keras.layers.Dense(hidden_size, activation='relu'),
                keras.layers.Dense(hidden_size // 2, activation='relu'),
                keras.layers.Dense(1)  # single output
            ])
        else:  # CNN
            inputs = keras.Input(shape=X_train.shape[1:])
            x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(hidden_size // 2, activation='relu')(x)
            outputs = keras.layers.Dense(1)(x)
            model = keras.Model(inputs, outputs)

        # Compiling with current learning rate
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )

        # Training and evaluating on validation set
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=8,
            batch_size=batch_size,
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]

        print(f"{task_name} | {model_type} | lr={learning_rate}, bs={batch_size}, units={hidden_size} -> val_loss={val_loss:.4f}")

        # The best config
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'units': hidden_size
            }

    return best_params, best_loss


def main():
    # hyperparameter grid
    search_grid = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [16, 32],
        'units': [64, 128]
    }

    tasks = ['taskA', 'taskB', 'taskC', 'taskD', 'taskE']
    model_types = ['MLP', 'CNN']

    overall_best = {}

    for task in tasks:
        print(f"\n>>> Tuning for {task}")
        overall_best[task] = {}
        for mtype in model_types:
            print(f"  - Model: {mtype}")
            best_cfg, best_loss = tune_hyperparameters(task, mtype, search_grid)
            print(f"    * Best {mtype}: {best_cfg}, val_loss={best_loss:.4f}\n")
            overall_best[task][mtype] = best_cfg


    with open('best_hyperparams.json', 'w') as outfile:
        json.dump(overall_best, outfile, indent=4)
    print("Hyperparameter search complete. Results written to best_hyperparams.json")


if __name__ == '__main__':
    main()
