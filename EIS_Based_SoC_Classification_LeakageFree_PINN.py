# ===============================================
# Physics-Informed ML for Battery SoC
# 2-Class Classification + Threshold Tuning
# ===============================================

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

# ----------------------------
# Load Dataset
# ----------------------------
impedance_path = r"C:\Users\Amruthamsha P Raju\Downloads\archive (3)\impedance.csv"
freq_path = r"C:\Users\Amruthamsha P Raju\Downloads\archive (3)\frequencies.csv"

impedance_df = pd.read_csv(impedance_path)
freq_df = pd.read_csv(freq_path)
data = impedance_df.merge(freq_df, on="FREQUENCY_ID")

# ----------------------------
# Robust Impedance Parsing
# ----------------------------
def parse_impedance(z):
    try:
        z = z.replace(" ", "").replace("(", "").replace(")", "")
        return complex(z)
    except:
        return np.nan

Z = data["IMPEDANCE_VALUE"].astype(str).apply(parse_impedance)
data["ReZ"] = np.real(Z)
data["ImZ"] = np.imag(Z)
data["|Z|"] = np.sqrt(data["ReZ"]**2 + data["ImZ"]**2)

data = data.dropna(subset=["ReZ", "ImZ", "|Z|", "SOC"]).reset_index(drop=True)

# ----------------------------
# EIS-Specific Features
# ----------------------------
data["LOG_FREQ"] = np.log10(data["FREQUENCY_VALUE"])
data["PHASE"] = np.arctan2(data["ImZ"], data["ReZ"])

# Chronological ordering
data = data.sort_values("FREQUENCY_VALUE").reset_index(drop=True)

# ----------------------------
# 2-Class SoC Binning
# ----------------------------
def soc_to_class(soc):
    return 0 if soc < 60 else 1  # Low–Mid vs High

data["SOC_CLASS"] = data["SOC"].apply(soc_to_class)

# ----------------------------
# Sliding Frequency Window (size = 3)
# ----------------------------
base_features = ["LOG_FREQ", "ReZ", "ImZ", "|Z|", "PHASE"]
X_list, y_list = [], []

for i in range(1, len(data) - 1):
    window_feats = data.iloc[i-1:i+2][base_features].values.flatten()
    X_list.append(window_feats)
    y_list.append(data.iloc[i]["SOC_CLASS"])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=int)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ----------------------------
# Chronological Split
# ----------------------------
n = len(X)
train_end = int(0.7 * n)
test_start = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]

# ----------------------------
# Class Weights
# ----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# =================================================
# Physics-Informed Neural Network (2-Class)
# =================================================
def build_classifier(input_dim):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="tanh",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(128, activation="tanh",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(64, activation="tanh"),
        layers.Dense(2, activation="softmax")
    ])

pinn = build_classifier(X_train.shape[1])
pinn.compile(
    optimizer=optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

pinn.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=0
)

# ----------------------------
# PINN Evaluation (THRESHOLD TUNING)
# ----------------------------
y_prob = pinn.predict(X_test, verbose=0)[:, 1]

THRESHOLD = 0.65
y_pred_pinn = (y_prob > THRESHOLD).astype(int)

acc_pinn = accuracy_score(y_test, y_pred_pinn)
f1_pinn = f1_score(y_test, y_pred_pinn)

print("\nPhysics-Informed NN (2-Class, Threshold Tuned)")
print(f"Threshold : {THRESHOLD}")
print(f"Accuracy  : {acc_pinn:.4f}")
print(f"F1 Score  : {f1_pinn:.4f}")
print(classification_report(y_test, y_pred_pinn, zero_division=0))

# =================================================
# Classical Models (for comparison)
# =================================================
models_dict = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}

print("\nClassical Model Performance:")

for name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name:20s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
