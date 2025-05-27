import os
import numpy as np
from tensorflow import keras
from models import build_mlp, build_cnn

def load_data(task):
    # train/test .npz, add channel dim
    train = np.load(f"data/{task}/train.npz")
    test  = np.load(f"data/{task}/test.npz")
    X_train, y_train = train["X"], train["y"]
    X_test,  y_test  = test["X"],  test["y"]
    # reshape to (N,25,25,1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]
    return (X_train, y_train), (X_test, y_test)

def train_and_eval(task, model_fn, epochs=20, batch_size=32):
    (X_train, y_train), (X_test, y_test) = load_data(task)
    model = model_fn(input_shape=X_train.shape[1:])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"{task} | {model_fn.__name__} → Test MSE: {loss:.4f}, MAE: {mae:.4f}")
    # Save model and history
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    model.save(f"models/{task}_{model_fn.__name__}.h5")
    np.savez(f"results/{task}_{model_fn.__name__}.npz", **history.history)
    return loss, mae

if __name__ == "__main__":
    tasks = ['taskA','taskB','taskC','taskD','taskE']
    for task in tasks:
        for fn in (build_mlp, build_cnn):
            train_and_eval(task, fn)
