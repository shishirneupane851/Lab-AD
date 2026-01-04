import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Load the dataset(supervised learning)
df = pd.read_csv('housing_price_dataset.csv')

print("Dataset Information:")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nDescriptive Statistics:")
print(df.describe())
print(f"\nMissing Values:")
print(df.isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(15, 10))

# Correlation heatmap
plt.subplot(2, 3, 1)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

# Price distribution
plt.subplot(2, 3, 2)
sns.histplot(df['Price'], kde=True, bins=50)
plt.title('Price Distribution')
plt.xlabel('Price')

# Income vs Price
plt.subplot(2, 3, 3)
sns.scatterplot(data=df, x='Avg. Area Income', y='Price')
plt.title('Income vs Price')

# House Age vs Price
plt.subplot(2, 3, 4)
sns.scatterplot(data=df, x='Avg. Area House Age', y='Price')
plt.title('House Age vs Price')

# Rooms vs Price
plt.subplot(2, 3, 5)
sns.scatterplot(data=df, x='Avg. Area Number of Rooms', y='Price')
plt.title('Rooms vs Price')

# Population vs Price
plt.subplot(2, 3, 6)
sns.scatterplot(data=df, x='Area Population', y='Price')
plt.title('Population vs Price')

plt.tight_layout()
plt.show()

# Prepare data for modeling
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results[name] = {
        'Model': model,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2,
        'CV R² Mean': cv_mean,
        'CV R² Std': cv_std
    }

    print(f"\n{name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

# Create comparison dataframe
results_df = pd.DataFrame({
    'Model': [name for name in results.keys()],
    'R² Score': [results[name]['R² Score'] for name in results.keys()],
    'RMSE': [results[name]['RMSE'] for name in results.keys()],
    'MAE': [results[name]['MAE'] for name in results.keys()],
    'CV R² Mean': [results[name]['CV R² Mean'] for name in results.keys()]
})

# Sort by R² score
results_df = results_df.sort_values('R² Score', ascending=False)

print("\n" + "=" * 80)
print("MODEL COMPARISON (Sorted by R² Score):")
print("=" * 80)
print(results_df.to_string(index=False))

# Plot model comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# R² Score comparison
axes[0, 0].barh(results_df['Model'], results_df['R² Score'])
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('Model Comparison - R² Score')
axes[0, 0].grid(True, alpha=0.3)

# RMSE comparison
axes[0, 1].barh(results_df['Model'], results_df['RMSE'])
axes[0, 1].set_xlabel('RMSE')
axes[0, 1].set_title('Model Comparison - RMSE')
axes[0, 1].grid(True, alpha=0.3)

# MAE comparison
axes[1, 0].barh(results_df['Model'], results_df['MAE'])
axes[1, 0].set_xlabel('MAE')
axes[1, 0].set_title('Model Comparison - MAE')
axes[1, 0].grid(True, alpha=0.3)

# CV R² Score comparison
axes[1, 1].barh(results_df['Model'], results_df['CV R² Mean'])
axes[1, 1].set_xlabel('Cross-Validation R² Score')
axes[1, 1].set_title('Model Comparison - Cross-Validation R² Score')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Get the best model
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['Model']

print(f"\n{'=' * 80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'=' * 80}")

# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot predictions vs actual values for best model
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(12, 5))

# Scatter plot of predictions vs actual
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'{best_model_name} - Predictions vs Actual')
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title(f'{best_model_name} - Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Error distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.title(f'{best_model_name} - Error Distribution')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True, alpha=0.3)
plt.show()

# Print some sample predictions
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
sample_actual = y_test.iloc[sample_indices].values
sample_pred = y_pred_best[sample_indices]

print("\nSample Predictions:")
print("-" * 80)
print(f"{'Index':<10} {'Actual Price':<20} {'Predicted Price':<20} {'Error':<20} {'Error %':<10}")
print("-" * 80)

for idx, act, pred in zip(sample_indices, sample_actual, sample_pred):
    error = act - pred
    error_pct = (error / act) * 100
    print(f"{idx:<10} ${act:<19,.2f} ${pred:<19,.2f} ${error:<19,.2f} {error_pct:>8.2f}%")

print("-" * 80)


# Make a function for new predictions
def predict_house_price(model, scaler, avg_income, house_age, num_rooms, num_bedrooms, area_population):
    """
    Predict house price for new input features

    Parameters:
    -----------
    model: trained model
    scaler: fitted scaler
    avg_income: Average area income
    house_age: Average area house age
    num_rooms: Average area number of rooms
    num_bedrooms: Average area number of bedrooms
    area_population: Area population

    Returns:
    --------
    Predicted price
    """
    # Create feature array
    features = np.array([[avg_income, house_age, num_rooms, num_bedrooms, area_population]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    return prediction


# Example prediction
print("\n" + "=" * 80)
print("EXAMPLE PREDICTION:")
print("=" * 80)
example_prediction = predict_house_price(
    best_model, scaler,
    avg_income=75000,
    house_age=6.5,
    num_rooms=7.0,
    num_bedrooms=4.0,
    area_population=35000
)
print(f"Predicted Price: ${example_prediction:,.2f}")