import pandas as pd

def vlookup_like(csv_file1, csv_file2, output_file):
    # Load the CSV files into DataFrames
    key_column = 'Postcode'  # The common column name to merge on
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Strip everything after a space in the key column of the first DataFrame
    df1[key_column] = df1[key_column].str.split(' ').str[0]
    
    # Drop rows with any missing values

    # Perform a merge operation based on the key column
    result = pd.merge(df1, df2, on=key_column, how='left')  # Use 'left' for left join, 'inner' for inner join
    # Drop rows with any missing values
    result.dropna(inplace=True)
    # Drop specified columns
    result.drop(columns=key_column, inplace=True)
    # Save the result to a new CSV file
    result.to_csv(output_file, index=False)

if __name__ == "__main__":
    csv_file1 = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/dropped.csv'  # Path to your first CSV file
    csv_file2 = '/Users/user/Documents/UCL work/Final Dissertation/District Data/Final Postcode Data.csv'  # Path to your second CSV file
    output_file = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/district_combined.csv'  # The output CSV file
    
    vlookup_like(csv_file1, csv_file2, output_file)
