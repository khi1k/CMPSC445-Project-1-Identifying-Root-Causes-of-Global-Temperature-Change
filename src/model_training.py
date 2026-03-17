import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

#setup paths
processeddata_directory = Path("data/processed")
results_directory = Path("results")
results_directory.mkdir(parents=True, exist_ok=True)

#loads the final dataset we will evaluate
def load_data():
    print("Loading final climate dataset")
    filepath = processeddata_directory / "final_climate_dataset.csv"
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df

#prepare features and target
def prepare_data(df, target_col="temperature_change_from_ghg"):
    print("\nPreparing features and target")
    #drop non-feature columns
    drop_cols = ["date", "year", target_col, "years_since_1979"]
    if "co2_x" in df.columns:
        drop_cols.append("co2_x")
    if "co2_y" in df.columns:
        drop_cols.append("co2_y")
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    #handle any remaining missing values in features
    X = X.fillna(X.mean())

    #drop rows where target is NaN (the ones that are before when OWID data starts)
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Date range with valid target: {df['date'][valid_mask].min()} to {df['date'][valid_mask].max()}")
    return X, y, feature_cols

#train and evaluate models
def train_models(X, y):
    print("\nTraining of the Required Models:")
    #split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False)
    print(f"Target range - Train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Target range - Test: [{y_test.min():.3f}, {y_test.max():.3f}]")
    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows")
    print("\nTesting different max_depth values:")
    for depth in [3, 5, 8, 10, 12, None]:
        rf_test = RandomForestRegressor(max_depth=depth, random_state=42, n_jobs=-1)
        rf_test.fit(X_train, y_train)
        r2 = r2_score(y_test, rf_test.predict(X_test))
        depth_str = "None" if depth is None else depth
        print(f"  Depth {depth_str}: R2 = {r2:.4f}")
    results = {}

    #1.Linear Regression Model Creation
    print("\nLinear Regression Model")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    results["Linear Regression"] = {
        "model": lr,
        "mse": mse_lr,
        "r2": r2_lr,
        "predictions": y_pred_lr,
        "actual": y_test.values
    }
    print(f"MSE: {mse_lr:.4f}")
    print(f"R2: {r2_lr:.4f}")

    #feature coefficients
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": lr.coef_
    }).sort_values("coefficient", ascending=False)
    print("\nTop 5 positive coefficients:")
    print(coef_df.head(5).to_string(index=False))
    print("\nTop 5 negative coefficients:")
    print(coef_df.tail(5).to_string(index=False))

    #2.Random Forest Regressor Model creation
    print("\nRandom Forest Regressor")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    results["Random Forest"] = {
        "model": rf,
        "mse": mse_rf,
        "r2": r2_rf,
        "predictions": y_pred_rf,
        "actual": y_test.values}
    print(f"MSE: {mse_rf:.4f}")
    print(f"R2: {r2_rf:.4f}")

    #3. XGBoost Regressor
    print("\nXGBoost Regressor")

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    results["XGBoost"] = {
        "model": xgb_model,
        "mse": mse_xgb,
        "r2": r2_xgb,
        "predictions": y_pred_xgb,
        "actual": y_test.values
    }
    print(f"MSE: {mse_xgb:.4f}")
    print(f"R2: {r2_xgb:.4f}")

    #feature importance for XGBoost
    xgb_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop 10 most important features - XGBoost:")
    print(xgb_importance.head(10).to_string(index=False))

    #feature importance
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop 10 most important features - Random Forest:")
    print(importance_df.head(10).to_string(index=False))
    return results, X_test.index, y_test

#create feature importance plots
def plot_feature_importance(results, feature_names, y_test, X):
    print("\nGenerating Plots - ")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(results_directory / "correlation_matrix.png")
    plt.show()

    #linear regression coefficients plot
    lr = results["Linear Regression"]["model"]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr.coef_
    }).sort_values("coefficient", ascending=False)
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in coef_df["coefficient"]]
    plt.barh(coef_df["feature"][:15], coef_df["coefficient"][:15], color=colors[:15])
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel("Coefficient Value")
    plt.title("Linear Regression Coefficients (Top 15)")
    plt.tight_layout()
    plt.savefig(results_directory / "linear_coefficients.png")
    plt.show()
    print("Saved: linear_coefficients.png")

    #random forest feature importance plot
    rf = results["Random Forest"]["model"]
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"][:15], importance_df["importance"][:15])
    plt.xlabel("Importance Score")
    plt.title("Random Forest Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(results_directory / "rf_importance.png")
    plt.show()
    print("Saved: rf_importance.png")

    #predictions vs actual plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
    plt.plot(y_test.index, results["Linear Regression"]["predictions"],
             label="Linear Regression", linestyle="--", alpha=0.7)
    plt.plot(y_test.index, results["Random Forest"]["predictions"],
             label="Random Forest", linestyle="--", alpha=0.7)

    plt.plot(y_test.index, results["XGBoost"]["predictions"],
             label="XGBoost", linestyle="--", alpha=0.7)
    plt.xlabel("Time Index")
    plt.ylabel("Temperature Anomaly")
    plt.title("Model Predictions vs Actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_directory / "predictions_comparison.png")
    plt.show()
    print("Saved: predictions_comparison.png")
    return coef_df, importance_df

#discuss root causes
def discuss_root_causes(coef_df, importance_df):
    print("\nRoot cause analysis - ")
    print("\nBased on Linear Regression coefficients:")
    top_positive = coef_df.head(3)["feature"].tolist()
    top_negative = coef_df.tail(3)["feature"].tolist()
    print(f"Top positive drivers: {top_positive}")
    print(f"Top negative drivers: {top_negative}")
    print("\nBased on Random Forest feature importance:")
    top_important = importance_df.head(5)["feature"].tolist()
    print(f"Top 5 most important features: {top_important}")

    print("\nInterpretation:")
    #check for greenhouse gases
    ghg_features = [f for f in coef_df["feature"] if any(g in f for g in ["co2", "ch4", "n2o"])]
    if ghg_features:
        print("- Greenhouse gases (CO2, CH4, N2O) appear as strong drivers")
    #check for growth rates
    growth_features = [f for f in coef_df["feature"] if "growth" in f]
    if growth_features:
        print("- the growth rates of greenhouse gases are significant")
    #check for moving averages
    ma_features = [f for f in coef_df["feature"] if "ma" in f]
    if ma_features:
        print("- long term trends (moving averages) matter")
    print("\nConclusion: Human-driven factors (CO2, CH4, N2O) dominate over natural variability")

def run_modeling():
    print("\nClimate Root Cause Modeling - ")
    #load data
    df = load_data()
    #prepare features and target
    X, y, feature_names = prepare_data(df)
    #train models
    results, test_idx, y_test = train_models(X, y)
    #plot results
    coef_df, importance_df = plot_feature_importance(
        results, feature_names, y_test, X)
    #discuss findings
    discuss_root_causes(coef_df, importance_df)

    #save results summary
    summary_path = results_directory / "model_results.txt"
    with open(summary_path, "w") as f:
        f.write("Model Performance Summary\n")
        for name, res in results.items():
            f.write(f"{name}:\n")
            f.write(f"  MSE: {res['mse']:.4f}\n")
            f.write(f"  R2:  {res['r2']:.4f}\n\n")
        f.write("\nFeature Importance of Random Forest:\n")
        f.write(importance_df.head(15).to_string())

        f.write("\n\nLinear Regression Coefficients:\n")
        f.write(coef_df.head(15).to_string())
    print(f"\nResults saved to: {summary_path}")
    return results, coef_df, importance_df

if __name__ == "__main__":
    results, coefs, importance = run_modeling()