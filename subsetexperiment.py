import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from models import build_mlp, build_cnn
from train import load_data


def run_subset_experiments():
    with open('best_hyperparams.json', 'r') as f:
        best_configs = json.load(f)

    # Fraction of data defined
    fractions = [0.25, 0.5, 1.0]
    records = []  # will be holding our results

    # Iterating over task
    for task, models_cfg in best_configs.items():
        print(f"\n--- Experimenting on {task} ---")
        (X_full, y_full), (X_test, y_test) = load_data(task)

        for model_name, cfg in models_cfg.items():
            print(f"\nModel: {model_name} with best params {cfg}")

            for frac in fractions:
                # Training data subset preparing
                if frac < 1.0:
                    X_train, _, y_train, _ = train_test_split(
                        X_full, y_full,
                        train_size=frac,
                        random_state=42
                    )
                else:
                    X_train, y_train = X_full, y_full

                # MLP model building
                if model_name == 'MLP':
                    model = build_mlp(input_shape=X_train.shape[1:])
                else:
                    model = build_cnn(input_shape=X_train.shape[1:])

                # Compiling with the tuned learning rate
                model.compile(
                    optimizer=Adam(cfg['learning_rate']),
                    loss='mse',
                    metrics=['mae']
                )

                # Training for the standard number of epochs
                model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=cfg['batch_size'],
                    verbose=0
                )

                # Evaluating on the test set
                mse, mae = model.evaluate(X_test, y_test, verbose=0)
                print(f"{int(frac*100)}% training → MSE: {mse:.3f}, MAE: {mae:.3f}")

                records.append({
                    'Task': task,
                    'Model': model_name,
                    'DataFraction': f"{int(frac*100)}%",
                    'Test_MSE': mse,
                    'Test_MAE': mae
                })

    # DataFrame
    df = pd.DataFrame(records)
    df.to_csv('subset_results.csv', index=False)
    print("\nAll experiments complete. Results saved to subset_results.csv.")

    # Pivot table
    pivot = df.pivot_table(
        index=['Task', 'Model'],
        columns='DataFraction',
        values=['Test_MSE', 'Test_MAE']
    )
    print("\nPivot table of Test MSE & MAE:")
    print(pivot)


if __name__ == '__main__':
    run_subset_experiments()
