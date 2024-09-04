import pandas as pd

def one_hot_encode_columns(input_csv, output_csv, columns_to_encode):
    """
    This function reads a CSV file, performs one-hot encoding on the specified columns, 
    and saves the resulting DataFrame to a new CSV file.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file where the result will be saved.
    - columns_to_encode (list of str): List of column names to one-hot encode.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_csv)
        
        # Check if the specified columns exist in the DataFrame
        for column in columns_to_encode:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in the CSV file.")
        
        # Perform one-hot encoding on the specified columns
        df = pd.get_dummies(df, columns=columns_to_encode)
        # Convert all values to numeric, errors='coerce' will convert non-convertible values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_csv, index=False)
        print(f"One-hot encoded columns '{columns_to_encode}' and saved the result to '{output_csv}'.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example of how this function can be used in another application
if __name__ == "__main__":
    input_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/macro_combined.csv'  # Path to the input CSV file
    output_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/final_processed.csv'  # Path to the output CSV file
    columns_to_encode = ['Type', 'Tenure', 'New_build']  # Columns to one-hot encode

    one_hot_encode_columns(input_csv, output_csv, columns_to_encode)
