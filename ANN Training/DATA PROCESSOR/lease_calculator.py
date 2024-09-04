import pandas as pd
import re
from datetime import datetime, timedelta
from dateutil.parser import parse

def process_lease_term(row):
    lease_term = row['Lease_Term']
    tenure = row['Tenure']
    
    # Check if the Tenure is 'Freehold'
    if tenure == 'Freehold':
        return 9999
    
    # Check if lease_term is not a string (e.g., NaN values)
    if not isinstance(lease_term, str):
        return None
    
    # Extract all dates from the string
    dates = re.findall(r'\d{1,2} [A-Za-z]+ \d{4}|\d{1,2}\.\d{1,2}\.\d{4}', lease_term)
    # Extract all durations from the string
    durations = re.findall(r'\d+ years', lease_term)
    
    # Function to convert found date to datetime object
    def convert_to_date(date_str):
        try:
            return parse(date_str, dayfirst=True)
        except ValueError:
            return None
    
    # Parse all extracted dates
    parsed_dates = [convert_to_date(date) for date in dates]
    
    # Filter out invalid dates
    parsed_dates = [date for date in parsed_dates if date is not None]
    
    # Determine the initial date to use
    target_date = None
    for date in parsed_dates:
        if date and date < datetime(2024, 6, 30):
            target_date = date
            break
    if not target_date and parsed_dates:
        target_date = parsed_dates[0]

    # If there's no valid target date, return None
    if not target_date:
        return None

    # Determine the number of years to add
    years_to_add = 0
    if durations:
        years_to_add = int(durations[0].split()[0])
    
    # Calculate the final date
    # Calculate the final date
    try:
        final_date = target_date + timedelta(days=365 * years_to_add)
    except OverflowError:
        # Handle the overflow error
        return f"Resulting date is out of range for adding {years_to_add} years to {target_date}"
    
    lease_year= final_date.year - 2024
    return lease_year

def epc_date_to_year(epc_date):
    # Convert the string to a datetime object
    epc_date = pd.to_datetime(epc_date)
    epc_year = epc_date.year - 2024
    return epc_year

def epc_rating_to_number(rating):
    rating_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    return rating_dict.get(rating, None)

def process_year_built(built_year):
    """Process the year built string to extract a single year."""
    if not built_year or isinstance(built_year, float) and pd.isna(built_year):
        return 'N/a'
    
    built_year_cleaned = re.sub(r'[^\d-]', '', built_year)
    
    try:
        if "-" in built_year_cleaned:
            start_year, end_year = map(int, built_year_cleaned.split('-'))
            house_age = 2024-round((start_year + end_year) / 2)
            return house_age
        else:
            house_age = 2024 - int(built_year_cleaned)
            return house_age
    except ValueError:
        return 'N/A'

# Function to read input CSV and process it
def process_lease_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Process Lease_Term column
    df['Processed_Lease_Term'] = df.apply(process_lease_term, axis=1)
    
    # Convert EPC ratings
    df['EPC_Rating_Current_Num'] = df['EPC_Rating_Current'].apply(epc_rating_to_number)
    df['EPC_Rating_Potential_Num'] = df['EPC_Rating_Potential'].apply(epc_rating_to_number)
    df['EPC_Year'] = df['EPC_Date'].apply(epc_date_to_year)
    
    
    # Process Year_Built column
    df['Processed_Year_Built'] = df['Year_Built'].apply(process_year_built)
    
    return df

# Example usage
if __name__ == "__main__":
    # Example usage
    input_path = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/sampled.csv'
    processed_df = process_lease_csv(input_path)
    
    # Display the processed DataFrame
    print(processed_df)
    
    # Save the processed DataFrame to a new CSV file if needed
    processed_df.to_csv('/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/lease_sampled.csv', index=False)
    



