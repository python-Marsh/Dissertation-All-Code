import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to clean and convert all columns to numeric
# def clean_all_columns(df):
#     for column in df.columns:
#         if column == 'Postcode':
#             continue  # Skip cleaning for postcode column
#         # Handle columns with '%' by removing the '%' and converting to float
#         if df[column].dtype == 'object' and df[column].str.contains('%').any():
#             df[column] = df[column].str.replace('%', '').astype(float)
#         else:
#             df[column] = pd.to_numeric(df[column].astype(str).str.replace(',', ''), errors='coerce')
#     return df


# Function to perform the tasks
def process_dataset(file_path, col_x, col_y):
    # Load the dataset
    
    # Convert all columns to numeric values
    # df = clean_all_columns(df)
    
    # 1. Count the number of missing values in the dataset
    missing_values_count = df.isnull().sum().sum()
    print(f"Total number of missing values in the dataset: {missing_values_count}")
    
    # Prepare the data for linear regression
    df_regression = df[[col_x, col_y]].dropna()
    
    # 2. Perform linear regression between the two columns
    X = df_regression[[col_x]].values
    y = df_regression[col_y].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Coefficients of the linear regression model
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Linear regression formula: {col_y} = {slope} * {col_x} + {intercept}")
    
    # 3. Fill in the missing values in col_y using the linear regression formula and input of col_x
    def fill_missing_values(row):
        if pd.isnull(row[col_y]):
            return int(slope * row[col_x] + intercept)
        else:
            return row[col_y]
    
    df[col_y] = df.apply(fill_missing_values, axis=1)
    
    # Save the updated dataframe to a new CSV file
    print("Missing values in column '{col_y}' have been filled and the updated dataset has been saved to 'updated_dataset.csv'.")

# Example usage
file_path = '/Users/user/Documents/UCL work/Final Dissertation/District Data/Final_Postcode_List_2.csv'  # Replace with the path to your dataset
df = pd.read_csv(file_path)
col_x = 'Population'  # Replace with the name of the column to use as predictor
col_y = 'Sales per month'  # Replace with the name of the column to fill missing values
process_dataset(file_path, col_x, col_y)
col_y = 'Turnover (sale)'  # Replace with the name of the column to fill missing values
process_dataset(file_path, col_x, col_y)
col_y = 'Population growth (10 year)'
process_dataset(file_path, col_x, col_y)
col_y = '5yr Price Growth'
process_dataset(file_path, col_x, col_y)
col_y = 'Crime rate'
process_dataset(file_path, col_x, col_y)
col_x = 'Density'  # Replace with the name of the column to use as predictor
col_y = 'Acres of green space / 1000 residents'  # Replace with the name of the column to fill missing values
process_dataset(file_path, col_x, col_y)
col_x = 'Avg. household income'  # Replace with the name of the column to use as predictor
col_y = df.columns[1]  # Replace with the name of the column to fill missing values
process_dataset(file_path, col_x, col_y)
df.to_csv('/Users/user/Documents/UCL work/Final Dissertation/District Data/updated_dataset.csv', index=False)
