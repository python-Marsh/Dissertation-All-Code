import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

# Assuming 'Processor.main' is a function that handles preprocessing
from Processor import main

# # comment when needed
n = 100000
columns_to_drop = [ 'Address','Bedrooms', 'URL','Price_paid', 'Lease_Term', 'EPC_Rating_Current', 'EPC_Rating_Potential', 'Year_Built', 'EPC_Date']
columns_to_encode = ['Type', 'Tenure', 'New_build']  # Columns to one-hot encode
main(n, columns_to_drop, columns_to_encode)

# Load the dataset
file_path = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/final_processed.csv'
data = pd.read_csv(file_path)
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

column_index = 1

# Extract the relevant mean and standard deviation values for the specific column
y_mean_val = scaler.mean_[column_index]
y_std_val = scaler.scale_[column_index]


# Apply Winsorization to cap the top and bottom 5% of all numeric columns
data_winsorized = data.copy()
numeric_columns = data.select_dtypes(include=[np.number]).columns
# Winsorize each numeric column
for column in numeric_columns:
    data_winsorized[column] = winsorize(data[column], limits=[0.05, 0.05])

# Set the target variable
target = 'price_per_sqr'

# Split data into features and target
X = data.drop(columns=[target ])
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
original_X_train = X_train.copy()

# Train the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Step 1: Create a mapping of features to groups
group_mapping = {
    # 'Lat': 'Coordinates',
    # 'Lng': 'Coordinates',
    'New_build_New-build': 'Building Age',
    'New_build_Old stock': 'Building Age',
    'Processed_Year_Built': 'Building Age',
    'Sq_feet': 'Size',
    'Degree': 'Education',
    'Density':'Density',
    'Acres of green space / 1000 residents': 'Green Area',
    'Type_Semi-detached house': 'House Type', 
    'Type_Flat': 'House Type',
    'Type_Terraced house': 'House Type',
    'Type_Detached house': 'House Type',
    'EPC_Year': 'EPC Certificate',
    'EPC_Rating_Current_Num': 'EPC Certificate',
    'EPC_Rating_Potential_Num': 'EPC Certificate',
    'Social rent': 'Social rent',
    # 'Average Income': 'Average Income',
    'earnings': 'Macroeconomic',
    'unemployment': 'Macroeconomic',
    'GDP index': 'Macroeconomic',
    '3 year mort fix': 'Macroeconomic',
    'Crime rate':'Crime rate',
    '10 yr GILT': 'Macroeconomic',
    'Turnover (sale)': 'Sales Volume', 
    'Inflation Rate': 'Macroeconomic',
    'Population': 'Regional Demographic',
    'Households': 'Regional Demographic',
    'Habitable_Rooms': 'Number of Rooms',
    'Processed_Lease_Term': 'Tenure Status',
    'Tenure_Leasehold':'Tenure Status',
    'Tenure_Freehold': 'Tenure Status',
    'Turnover (sale)': 'Sales Volume', 
    'Inflation Rate': 'Macroeconomic',
    'Deprivation': 'Deprivation',
    'State Schools': 'State Schools',
    '5yr Price Growth': 'Region Growth',
    'Population growth (10 year)': 'Regional Demographic',
    'Sales per month': 'Sales Volume'
}




# Ensure all features are correctly mapped and included
missing_features = []
for feature, group in group_mapping.items():
    if feature not in features:
        missing_features.append(feature)
        print(f"Feature '{feature}' is missing from the model's features.")

# If there are missing features that should be included:
if missing_features:
    print(f"The following features are missing and might affect the importance distribution: {missing_features}")

# Step 2: Create a new dictionary to hold grouped importances
grouped_importances_dict = {group: 0 for group in set(group_mapping.values())}

# Add importances to the groups
for feature, group in group_mapping.items():
    if feature in X.columns:
        idx = X.columns.get_loc(feature)
        grouped_importances_dict[group] += importances[idx]
    else:
        print(f"Feature '{feature}' not found in the model's features. Skipping.")

# Convert the dictionary to a pandas Series
grouped_importances = pd.Series(grouped_importances_dict)

# Step 3: Sort the grouped importances
grouped_importances = grouped_importances.sort_values(ascending=False)

# Normalize the importances to sum to 1
normalized_importances = grouped_importances / grouped_importances.sum()

# Plot the normalized grouped feature importances
plt.figure(figsize=(10, 6))
plt.title('Normalized Grouped Feature Importances')
plt.bar(range(len(normalized_importances)), normalized_importances, align='center')
plt.xticks(range(len(normalized_importances)), normalized_importances.index, rotation=90)
plt.xlabel('Group')
plt.ylabel('Normalized Importance')
plt.tight_layout()
plt.show()

# Step 1: Create a mapping of features to groups
group_mapping = {
    'Lat': 'Individual Factor',
    'Lng': 'Individual Factor',
    'New_build_New-build': 'Individual Factor',
    'New_build_Old stock': 'Individual Factor',
    'Processed_Year_Built': 'Individual Factor',
    'Sq_feet': 'Individual Factor',
    'Degree': 'Planning Factor',
    'Density':'Planning Factor',
    'Acres of green space / 1000 residents': 'Planning Factor',
    'Type_Semi-detached house': 'Individual Factor', 
    'Type_Flat': 'Individual Factor',
    'Type_Terraced house': 'Individual Factor',
    'Type_Detached house': 'Individual Factor',
    'EPC_Year': 'Individual Factor',
    'EPC_Rating_Current_Num': 'Individual Factor',
    'EPC_Rating_Potential_Num': 'Individual Factor',
    'Social rent': 'Planning Factor',
    'Average Income': 'Planning Factor',
    'earnings': 'Macroeconomic',
    'unemployment': 'Macroeconomic',
    'GDP index': 'Macroeconomic',
    '3 year mort fix': 'Macroeconomic',
    'Crime rate':'Planning Factor',
    '10 yr GILT': 'Macroeconomic',
    'Turnover (sale)': 'Planning Factor', 
    'Inflation Rate': 'Macroeconomic',
    'Population': 'Planning Factor',
    'Households': 'Planning Factor',
    'Habitable_Rooms': 'Individual Factor',
    'Processed_Lease_Term': 'Individual Factor',
    'Tenure_Leasehold':'Individual Factor',
    'Tenure_Freehold': 'Individual Factor',
    'Inflation Rate': 'Macroeconomic',
    'Deprivation': 'Planning Factor',
    'State Schools': 'Planning Factor',
    '5yr Price Growth': 'Planning Factor',
    'Population growth (10 year)': 'Planning Factor',
    'Sales per month': 'Planning Factor'
}


# Ensure all features are correctly mapped and included
missing_features = []
for feature, group in group_mapping.items():
    if feature not in features:
        missing_features.append(feature)
        print(f"Feature '{feature}' is missing from the model's features.")

# If there are missing features that should be included:
if missing_features:
    print(f"The following features are missing and might affect the importance distribution: {missing_features}")

# Step 2: Create a new dictionary to hold grouped importances
grouped_importances_dict = {group: 0 for group in set(group_mapping.values())}

# Add importances to the groups
for feature, group in group_mapping.items():
    if feature in X.columns:
        idx = X.columns.get_loc(feature)
        grouped_importances_dict[group] += importances[idx]
    else:
        print(f"Feature '{feature}' not found in the model's features. Skipping.")

# Convert the dictionary to a pandas Series
grouped_importances = pd.Series(grouped_importances_dict)

# Step 3: Sort the grouped importances
grouped_importances = grouped_importances.sort_values(ascending=False)

# Normalize the importances to sum to 1
normalized_importances = grouped_importances / grouped_importances.sum()

# Plot the normalized grouped feature importances
plt.figure(figsize=(10, 6))
plt.title('Normalized Grouped Feature Importances')
plt.bar(range(len(normalized_importances)), normalized_importances, align='center')
plt.xticks(range(len(normalized_importances)), normalized_importances.index, rotation=90)
plt.xlabel('Group')
plt.ylabel('Normalized Importance')
plt.tight_layout()
plt.show()

# Predict on the test set
y_pred = rf.predict(X_test)

# Ensure 1D arrays
y_test = y_test.flatten() if len(y_test.shape) > 1 else y_test
y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

# Predicted vs Actual Values Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='b', label=f'Data Points: {len(y_test)}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r')
plt.title('Residuals Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Correlation Matrix with Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Predicted vs Actual Values Histogram
plt.figure(figsize=(10, 6))
plt.hist([y_test, y_pred], label=['Actual Values', 'Predicted Values'], alpha=0.7, bins=30)
plt.title('Histogram of Actual vs Predicted Values')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Cumulative Feature Importances (Random Forest)
cumulative_importances = np.cumsum(importances[indices])
plt.figure(figsize=(10, 6))
plt.plot(range(X.shape[1]), cumulative_importances, 'b-')
plt.hlines(y=0.95, xmin=0, xmax=X.shape[1], colors='r', linestyles='dashed')
plt.title('Cumulative Feature Importances')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.show()

# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
average_price = y.mean()

print(f'R-squared: {r2}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')
print(f'Average Price: {average_price}')

# Residual Analysis for RF
residuals_rf = y_test - rf.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(rf.predict(X_test), residuals_rf, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r')
plt.title('Residuals Plot (Random Forest)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Distribution of Residuals (Random Forest)
plt.figure(figsize=(10, 6))
sns.histplot(residuals_rf, kde=True, color='blue')
plt.title('Distribution of Residuals (Random Forest)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

from sklearn.ensemble import RandomForestRegressor

# Fit a surrogate RandomForest model
surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
surrogate_model.fit(original_X_train, y_train)

# Define the features for which you want to plot the PDP
features_to_plot = list(range(original_X_train.shape[1]))  # Replace with indices of the features you're interested in

# Plot each feature's PDP separately
for feature in features_to_plot:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(surrogate_model, original_X_train, [feature], ax=ax)
    plt.suptitle(f'Partial Dependence Plot for Feature {feature}')
    plt.show()