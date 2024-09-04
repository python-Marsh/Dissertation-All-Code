import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

def get_nhs_distances(input_file, intermediate_file):
    driver = webdriver.Chrome()  # Adjust the path to your ChromeDriver if necessary
    df = pd.read_csv(input_file)  # Load the CSV file
    postcodes = df.iloc[:, 2].tolist()
    distances = []

    for i, postcode in enumerate(postcodes):
        url = 'https://www.nhs.uk/service-search/hospital/results/'
        driver.get(url + postcode)
        try:
            distance_NHS = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located((By.ID, 'distance_0'))
            )
            temp = distance_NHS.text
            match = re.search(r'\d+(\.\d+)?', temp)
            if match:
                distances.append(match.group())
            else:
                distances.append('N/A')
        except Exception as e:
            print(f"Error for postcode {postcode}: {e}")
            distances.append('Error')

        # Save intermediate results every 100 entries
        if (i + 1) % 100 == 0:
            save_intermediate_results(df.iloc[i+1-100:i+1], distances[-100:], intermediate_file)

    driver.quit()

    # Save final results
    save_intermediate_results(df.iloc[len(postcodes)-len(distances):], distances, intermediate_file)
    return distances

def save_intermediate_results(df_chunk, new_distances, intermediate_file):
    df_chunk['Hospital distance'] = new_distances
    if not os.path.exists(intermediate_file):
        df_chunk.to_csv(intermediate_file, index=False)
    else:
        df_chunk.to_csv(intermediate_file, mode='a', header=False, index=False)
    print(f"Intermediate results saved to {intermediate_file}")

# Example usage
if __name__ == "__main__":
    input_file = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/LetsPartyNashville.csv'  # Replace with your CSV file path
    intermediate_file = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/LetsPartyNashville_intermediate.csv'  # Replace with your intermediate file path

    # Get NHS distances
    nhs_distances = get_nhs_distances(input_file, intermediate_file)

    print(f"All distances have been processed and saved to {intermediate_file}")


    # # Specify the file path
    # file_path = '/Users/user/Desktop/NHS_Distances.csv'  # Replace with your desired file path

    # # Write the DataFrame to a CSV file
    # df.to_csv(file_path, index=False)
    # print(df)
    # print("CSV file has been written.")

