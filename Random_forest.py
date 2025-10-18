import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# === Load Data ===
df_train = pd.read_csv("C:\\Users\\Hp\\OneDrive\\Documents\\shell ai\\Main Content\\dataset\\train.csv")
df_test = pd.read_csv("C:\\Users\\Hp\\OneDrive\\Documents\\shell ai\\Main Content\\dataset\\test.csv")

# === Feature + Target Setup ===
features = df_train.columns[:55].tolist()
targets = df_train.columns[55:].tolist()

X_train_raw = df_train[features]
X_test_raw = df_test[features].iloc[:500]
Y_train = df_train[targets]

# === Impute + Scale ===
imp = SimpleImputer()
X_train = imp.fit_transform(X_train_raw)
X_test = imp.transform(X_test_raw)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

df_train_scaled = pd.DataFrame(X_train, columns=features)
df_test_scaled = pd.DataFrame(X_test, columns=features)

# === Custom RF iterations (n_estimators) per target ===
rf_param_grid = {
    'BlendProperty1': 800,
    'BlendProperty2': 300,
    'BlendProperty3': 450,
    'BlendProperty4': 350,
    'BlendProperty5': 400,
    'BlendProperty6': 500,
    'BlendProperty7': 300,
    'BlendProperty8': 450,
    'BlendProperty9': 350,
    'BlendProperty10': 300,
}

# === Model Loop with RF ===
final_preds = []
metrics_log = []

for col in targets:
    print(f"\n🌲 RandomForest Training for {col}")
    y_col = Y_train[col].values
    n_estimators = rf_param_grid.get(col, 300)

    # Feature selection using temp RF
    temp_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=5, random_state=42)
    temp_rf.fit(X_train, y_col)
    importances = temp_rf.feature_importances_
    important_features = [features[i] for i, imp in enumerate(importances) if imp > 0.001]

    if not important_features:
        print("⚠ No important features found. Using all features.")
        important_features = features

    X_train_sub = df_train_scaled[important_features].values
    X_test_sub = df_test_scaled[important_features].values

    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=12, random_state=42)
    rf_model.fit(X_train_sub, y_col)

    y_pred_test = rf_model.predict(X_test_sub)
    final_preds.append(y_pred_test)

    y_pred_train = rf_model.predict(X_train_sub)
    r2 = r2_score(y_col, y_pred_train)
    mape = mean_absolute_percentage_error(y_col, y_pred_train)
    metrics_log.append((col, r2, mape))
    print(f"{col} ✅ R²: {r2:.4f}, MAPE: {mape:.4f}")

# === Save Predictions ===
pred_df = pd.DataFrame(np.array(final_preds).T, columns=targets)
pred_df.insert(0, "ID", range(1, 501))
pred_df.to_csv("rf_predictions.csv", index=False)
print("\n📁 Predictions saved to rf_predictions.csv")

# === Summary ===
print("\n📊 Final Scores (Random Forest):")
for name, r2, mape in metrics_log:
    print(f"{name} - R²: {r2:.4f}, MAPE: {mape:.4f}")
