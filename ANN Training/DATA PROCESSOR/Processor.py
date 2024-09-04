from Random_Sample import random_sample
from lease_calculator import process_lease_csv
from drop_missing import process_csv
from district_combiner import vlookup_like
from macro_combiner import combine_csv_files
from one_hot import one_hot_encode_columns

def main(n, columns_to_drop, columns_to_encode):
    # Define the paths to the input and output files
    input_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/merged.csv'  # Input for random sampling
    sampled_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/sampled.csv'  # Output from random sampling
    leased_csv ='/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/lease_sampled.csv'
    processed_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/dropped.csv'  # Output from drop missing
    district_data_csv = '/Users/user/Documents/UCL work/Final Dissertation/District Data/Final Postcode Data.csv'  # Input for district combiner
    district_combined_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/district_combined.csv'  # Output from district combiner
    macro_combined_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/macro_combined.csv'  # Output from macro combiner
    final_output_csv = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/final_processed.csv'  # Final output after one-hot encoding

    # Step 1: Random sample the data
    print("Performing random sampling...")
    random_sample(input_csv, sampled_csv, n, random_state=42)
    
    lease_df = process_lease_csv(sampled_csv)
    lease_df.to_csv(leased_csv, index=False)
    print('leased calculated')

    # Step 2: Drop missing values from the sampled data
    print("Dropping missing values...")
    processed_df = process_csv(leased_csv, columns_to_drop)
    processed_df.to_csv(processed_csv, index=False)
    print('dropped file saved')

    # Step 3: Use processed data as input for district combiner
    print("Combining with district files...")
    vlookup_like(processed_csv, district_data_csv, district_combined_csv)

    # Step 4: Use district combiner’s output as input for macro combiner
    print("Combining with macroeconomic files...")
    combine_csv_files(district_combined_csv, macro_combined_csv)

    # Step 5: Perform one-hot encoding on the final output
    print("Performing one-hot encoding...")
    one_hot_encode_columns(macro_combined_csv, final_output_csv, columns_to_encode)
    print(f"Final output saved to {final_output_csv}")

if __name__ == "__main__":
    n = 1000000
    columns_to_drop = [ 'Address','Bedrooms', 'URL','Price_paid', 'Lease_Term', 'EPC_Rating_Current', 'EPC_Rating_Potential', 'Year_Built', 'EPC_Date']  # Replace with actual column names to drop
    columns_to_encode = ['Type', 'Tenure', 'New_build']  # Columns to one-hot encode
    main(n, columns_to_drop, columns_to_encode)
  