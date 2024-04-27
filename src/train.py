import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from azureml.core import Run

# Initialize Azure ML run
run = Run.get_context()

def remove_outliers(data, threshold=3):
    """
    Remove outliers from the data using z-scores.
    
    Parameters:
    - data: DataFrame, input data
    - threshold: int, z-score threshold
    
    Returns:
    - DataFrame, data with outliers removed
    """
    z_scores = (data.select_dtypes(include=['int64', 'float64']) - data.select_dtypes(include=['int64', 'float64']).mean()) / data.select_dtypes(include=['int64', 'float64']).std()
    outlier_rows = z_scores.abs().gt(threshold).any(axis=1)
    data = data[~outlier_rows]
    return data

def add_feature_engineering(data):
    """
    Add new features based on existing data.
    
    Parameters:
    - data: DataFrame, input data
    
    Returns:
    - DataFrame, data with new features added
    """
    data["bmi_category"] = pd.cut(data["bmi"], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])
    data["is_located_north"] = ((data["region"] == "northwest") | (data["region"] == "northeast")).astype(int)
    data["is_located_south"] = ((data["region"] == "southeast") | (data["region"] == "southwest")).astype(int)
    data["more_than_one_child"] = (data["children"] > 1).astype(int)
    data = data.drop(["children", "region", "bmi"], axis=1)
    return data

# Predict methods
def predict_charges_linear_regression(input_data):
    """Predict medical charges using Linear Regression."""
    # Encode categorical variables
    input_data_encoded = pd.get_dummies(input_data)
    missing_cols = set(X.columns) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[X.columns]
    charges_pred = model.predict(input_data_encoded)
    return charges_pred

def predict_charges_random_forest_regression(input_data):
    """Predict medical charges using Random Forest Regression."""
    input_data_encoded = pd.get_dummies(input_data)
    missing_cols = set(X.columns) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[X.columns]
    charges_pred = model_random_forest.predict(input_data_encoded)
    return charges_pred

# Load data
data = pd.read_csv("insurance.csv")
data = remove_outliers(data)
data_with_features = add_feature_engineering(data)
data_encoded = pd.get_dummies(data_with_features)
data_encoded.to_csv("data_encoded.csv", index=False)

# Split data into features and target variable
X = data_encoded.drop("charges", axis=1)
y = data_encoded["charges"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
model = LinearRegression()
model.fit(X_train, y_train)
model_random_forest = RandomForestRegressor()
model_random_forest.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)
y_pred_random_forest = model_random_forest.predict(X_val)

# Evaluation metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

mse_random_forest = mean_squared_error(y_val, y_pred_random_forest)
mae_random_forest = mean_absolute_error(y_val, y_pred_random_forest)
r2_random_forest = r2_score(y_val, y_pred_random_forest)

# Make a prediction
input_data = pd.DataFrame({
    "age": [25],
    "sex": ["male"],
    "bmi": [22.5],
    "children": [0],
    "smoker": ["yes"],
    "region": ["southwest"]
})
charges_pred_linear_regression = predict_charges_linear_regression(input_data)
charges_pred_random_forest = predict_charges_random_forest_regression(input_data)

# Log evaluation metrics
run.log('Linear Regression MSE', mse)
run.log('Linear Regression MAE', mae)
run.log('Linear Regression R2', r2)
run.log('Linear Regression Prediction', charges_pred_linear_regression[0])
run.log('Random Forest MSE', mse_random_forest)
run.log('Random Forest MAE', mae_random_forest)
run.log('Random Forest R2', r2_random_forest)
run.log('Random Forest Prediction', charges_pred_random_forest[0])

# Save plots and log to Azure ML
plt.figure(figsize=(10, 8))

# Count plot of gender
sns.countplot(data=data, x="sex", palette="viridis")
plt.title("Count of Gender")
plt.savefig("count_plot_gender.png")
run.log_image("Count Plot Gender", path="count_plot_gender.png")
plt.close()

# Count plot of smoker
sns.countplot(data=data, x="smoker", palette="viridis")
plt.title("Count of Smoker")
plt.savefig("count_plot_smoker.png")
run.log_image("Count Plot Smoker", path="count_plot_smoker.png")
plt.close()

# Count plot of children
sns.countplot(data=data, x="children", palette="viridis")
plt.title("Count of Children")
plt.xticks([int(x) for x in plt.xticks()[0]])  # Set integer tick labels on x-axis
plt.savefig("count_plot_children.png")
run.log_image("Count Plot Children", path="count_plot_children.png")
plt.close()

# Count plot of region
sns.countplot(data=data, x="region", palette="viridis")
plt.title("Count of Region")
plt.savefig("count_plot_region.png")
run.log_image("Count Plot Region", path="count_plot_region.png")
plt.close()

# Boxplot of residuals for Random Forest model
sns.boxplot(x=y_pred_random_forest - y_val, showmeans=True)
plt.title("Boxplot of Residuals (Random Forest)")
plt.savefig("boxplot_residuals_random_forest.png")
run.log_image("Boxplot Residuals (Random Forest)", path="boxplot_residuals_random_forest.png")
plt.close()

# Scatter plot of age vs. charges for smokers
sns.lmplot(data=data, x="age", y="charges", hue="smoker")
plt.title("Scatter plot of Age vs. Charges for Smokers")
plt.savefig("scatter_age_charges_smokers.png")
run.log_image("Scatter Age Charges Smokers", path="scatter_age_charges_smokers.png")
plt.close()
