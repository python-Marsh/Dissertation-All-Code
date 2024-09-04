import pandas as pd


def combine_csv_files(csv_files, output_file):
    # Function to parse dates with multiple formats
    def parse_dates(date_str):
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %I:%M:%S %p'):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                pass
        return pd.to_datetime(date_str)  # Fallback to default parsing
    
    # Load the first CSV file
    df1 = pd.read_csv('/Users/user/Documents/UCL work/Final Dissertation/Macroeconomic Data/combined_time_series.csv')
    
    # Load the second CSV file
    df2 = pd.read_csv(csv_files)
    
    # Convert the 'Date' columns to datetime format using the parse_dates function
    df1['Date'] = df1['Date'].apply(parse_dates)
    df2['Date'] = df2['Date'].apply(parse_dates)
    
    # Ensure both 'Date' columns are in datetime format
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    # Normalize df2's 'Date' to remove time part if necessary
    df2['Date'] = df2['Date'].dt.normalize()
    
    # Filter df1 to only include dates present in df2
    df1_filtered = df1[df1['Date'].isin(df2['Date'].unique())]
    
    # Merge the two dataframes on the 'Date' column
    merged_df = pd.merge(df2, df1_filtered, on='Date', how='left')
    # Drop rows with any missing values
    merged_df.dropna(inplace=True)
    merged_df.drop(columns='Date', inplace=True)
    
    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)
    
    print("Merged file saved as '/Users/user/Documents/UCL work/Final Dissertation/merged_file.csv'")

if __name__ == "__main__":
    output_file = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/macro_combined.csv'
    csv_files = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/district_combined.csv'
    combine_csv_files(csv_files, output_file)