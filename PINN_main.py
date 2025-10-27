# ===============================================
# Physics-Informed Machine Learning for Battery SoC
# Author: Amruthamsha P Raju
# ===============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ----------------------------
# Load Dataset
# ----------------------------
impedance_path = r"C:\Users\Amruthamsha P Raju\Downloads\archive (3)\impedance.csv"
freq_path = r"C:\Users\Amruthamsha P Raju\Downloads\archive (3)\frequencies.csv"

impedance_df = pd.read_csv(impedance_path)
freq_df = pd.read_csv(freq_path)

# Merge using FREQUENCY_ID
data = impedance_df.merge(freq_df, on="FREQUENCY_ID")

# Extract real and imaginary parts of impedance (ReZ, ImZ)
data["ReZ"] = data["IMPEDANCE_VALUE"].str.extract(r"\(([-+]?\d*\.\d+|\d+)")[0].astype(float)
data["ImZ"] = data["IMPEDANCE_VALUE"].str.extract(r"\+([-+]?\d*\.\d+|\d+)j")[0].astype(float)

# Impedance magnitude
data["|Z|"] = np.sqrt(data["ReZ"]**2 + data["ImZ"]**2)

# ----------------------------
# Features & Target
# ----------------------------
X = data[["FREQUENCY_VALUE", "ReZ", "ImZ", "|Z|"]].values
y = data["SOC"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Physics-Informed Loss Function
# ----------------------------
def physics_loss(y_true, y_pred, inputs):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # |Z| from inputs (4th column)
    Z_mag = inputs[:, 3:4]

    # Correlation between predicted SoC and |Z| (should be negative)
    corr = tf.reduce_mean(
        (y_pred - tf.reduce_mean(y_pred)) * (Z_mag - tf.reduce_mean(Z_mag))
    ) / (tf.math.reduce_std(y_pred) * tf.math.reduce_std(Z_mag) + 1e-8)
    corr_penalty = tf.square(tf.maximum(corr, 0.0))  # penalize positive correlation

    # Smoothness: reduce large changes across nearby points
    diff = y_pred[1:] - y_pred[:-1]
    smooth_penalty = tf.reduce_mean(tf.square(diff))

    return mse + 0.1 * corr_penalty + 0.01 * smooth_penalty


class PhysicsLoss(tf.keras.losses.Loss):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = tf.constant(inputs, dtype=tf.float32)
    def call(self, y_true, y_pred):
        return physics_loss(y_true, y_pred, self.inputs)


# ----------------------------
# Build Physics-Informed Neural Network
# ----------------------------
def build_PINN(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation="tanh", input_shape=(input_dim,)),
        layers.Dense(128, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(1)
    ])
    return model

pinn = build_PINN(X_train.shape[1])
pinn.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss=PhysicsLoss(X_train))

# ----------------------------
# Train PINN
# ----------------------------
print("Training Physics-Informed Neural Network...")
pinn.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

# ----------------------------
# Evaluate PINN
# ----------------------------
y_pred_pinn = pinn.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred_pinn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_pinn)

print("\nPhysics-Informed PINN Performance:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ----------------------------
# Classical ML Models for Comparison
# ----------------------------
models_dict = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Convert regression → classification for SoC level comparison
    y_test_class = (y_test > np.median(y_test)).astype(int)
    y_pred_class = (y_pred > np.median(y_pred)).astype(int)
    acc = accuracy_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)

    results[name] = [mse, rmse, r2, acc, f1]

# Add PINN results
y_test_class = (y_test > np.median(y_test)).astype(int)
y_pred_class = (y_pred_pinn > np.median(y_pred_pinn)).astype(int)
acc = accuracy_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
results["Physics-Informed NN"] = [mse, rmse, r2, acc, f1]

# ----------------------------
# Final Results Table
# ----------------------------
df_results = pd.DataFrame(results, index=["MSE", "RMSE", "R2", "Accuracy", "F1 Score"]).T
print("\nPerformance Metrics Summary:")
print(df_results)

#====================================================================================================================

