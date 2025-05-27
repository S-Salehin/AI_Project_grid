
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from train import load_data


file_name_map = {
    'MLP': 'build_mlp',
    'CNN': 'build_cnn'
}

# Best hyperparameters loaded
with open('best_hyperparams.json', 'r') as f:
    best_cfgs = json.load(f)

def plot_examples(task, model_name):
    """Plot 5 lowest-error and 5 highest-error test samples"""
    base = file_name_map[model_name]
    model_path = f"models/{task}_{base}.h5"
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    (_, _), (X_test, y_true) = load_data(task)
    y_pred = model.predict(X_test).flatten()
    errors = np.abs(y_pred - y_true)
    idx = np.argsort(errors)
    best_idx, worst_idx = idx[:5], idx[-5:]

    os.makedirs('figures', exist_ok=True)
    for indices, label in [(best_idx, '5_best'), (worst_idx, '5_worst')]:
        fig, axes = plt.subplots(1, len(indices), figsize=(len(indices)*2, 2))
        for ax, i in zip(axes, indices):
            ax.imshow(X_test[i, ..., 0], cmap='gray')
            ax.axis('off')
            ax.set_title(f"T:{y_true[i]:.2f}\nP:{y_pred[i]:.2f}")
        fig.suptitle(f"{task} | {model_name} | {label}")
        fig.savefig(f"figures/{task}_{model_name}_{label}.png", bbox_inches='tight')
        plt.close(fig)
    print(f"Saved best/worst examples for {task} {model_name}")

def plot_learning_curve(task, model_name):
    """Plot train vs val MSE over epochs"""
    base = file_name_map[model_name]
    fp = f"results/{task}_{base}.npz"
    print(f"Loading history from {fp}")
    data = np.load(fp)
    epochs = np.arange(1, len(data['loss'])+1)

    os.makedirs('figures', exist_ok=True)
    plt.figure()
    plt.plot(epochs, data['loss'],     label='Train MSE')
    plt.plot(epochs, data['val_loss'], label='Val MSE')
    plt.title(f"{task} | {model_name} Learning Curve")
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.savefig(f"figures/{task}_{model_name}_learning_curve.png", bbox_inches='tight')
    plt.close()
    print(f"Saved learning curve for {task} {model_name}")

def plot_scatter(task, model_name):
    """Plot true vs predicted with y=x."""
    base = file_name_map[model_name]
    model_path = f"models/{task}_{base}.h5"
    model = keras.models.load_model(model_path, compile=False)

    (_, _), (X_test, y_true) = load_data(task)
    y_pred = model.predict(X_test).flatten()
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(4,4))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.title(f"{task} | {model_name} True vs Pred")
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.savefig(f"figures/{task}_{model_name}_scatter.png", bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot for {task} {model_name}")

if __name__ == '__main__':
    for task, models in best_cfgs.items():
        for m in models.keys():
            plot_examples(task, m)
        for m in models.keys():
            plot_learning_curve(task, m)
        for m in models.keys():
            plot_scatter(task, m)
    print("Done—all figures are in the ./figures/ folder.")
