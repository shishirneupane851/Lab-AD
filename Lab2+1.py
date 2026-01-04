import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset(simple linear regression)                                  
# y--> dependent Variable
# x--> independent variable
data = pd.read_csv("housing_price_dataset.csv")

# Select feature and target
X = data[['Avg. Area Income']]
y = data['Price']

# Train-test spliting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# PLOT: Scatter + Regression Line

plt.figure(figsize=(10, 6))

# Scatter plot of actual data
plt.scatter(X, y, label="Actual Data", alpha=0.5)

# Regression line
plt.plot(X, model.predict(X), color='red', label="Regression Line", linewidth=2)

plt.xlabel("Avg. Area Income")
plt.ylabel("Price")
plt.title("Linear Regression: Avg. Area Income vs Price")
plt.legend()
plt.grid(True)
plt.show()