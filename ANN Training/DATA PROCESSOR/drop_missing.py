import pandas as pd

def process_csv(file_path, columns_to_drop):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Report existing columns and missing values
    print("Existing columns and missing values before any processing:")
    print(df.isnull().sum())
    print()

    # Drop specified columns
    df.drop(columns=columns_to_drop, inplace=True)

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Report existing columns and missing values after dropping columns and rows with missing values
    print("Existing columns and missing values after dropping columns and rows with missing values:")
    print(df.isnull().sum())
    print()


    # Return the processed DataFrame
    return df

# If you want to save the processed DataFrame to a new CSV file:
if __name__ == "__main__":
    print('not my business')
    file_path = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/lease_sampled.csv'  # Replace with the path to your CSV file
    columns_to_drop = ['Address','Bedrooms', 'URL','Price_paid', 'Lease_Term', 'EPC_Rating_Current', 'EPC_Rating_Potential', 'Year_Built', 'EPC_Date']  # Replace with actual column names to drop
    df = process_csv(file_path, columns_to_drop)
    df.to_csv('/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/dropped.csv', index=False)
