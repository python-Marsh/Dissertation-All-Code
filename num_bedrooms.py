import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

def process_year_built(built_year):
    """Process the year built string to extract a single year."""
    if not built_year or built_year.lower() == 'no data':
        return 'No Data'  # Return 'N/A' if the year built is empty or 'No Data'
    
    # Remove any non-numeric characters (e.g., words)
    built_year_cleaned = re.sub(r'[^\d-]', '', built_year)
    
    try:
        if "-" in built_year_cleaned:
            start_year, end_year = map(int, built_year_cleaned.split('-'))
            return round((start_year + end_year) / 2)
        else:
            return int(built_year_cleaned)
    except ValueError:
        return 'N/A'

def clean_address(text):
    return re.sub(r'Select a property to continue\.\s*', '', text).strip()

def split_address(address):
    """Splits an address into words and numbers."""
    words = re.findall(r'\b[A-Za-z]+\b', address)
    numbers = re.findall(r'\b\d+\b', address)
    return words, numbers

def compare_addresses(addr1, addr2):
    """Compares two addresses by their words and numbers."""
    words1, numbers1 = split_address(addr1)
    words2, numbers2 = split_address(addr2)

    # Check if the numbers match
    if numbers1 != numbers2:
        return False

    # Check if all words in the shorter address are in the longer address
    words_in_each_other = (all(word in words2 for word in words1) or
                           all(word in words1 for word in words2))
    
    return words_in_each_other


def extract_property_details(driver, link):
    print(f"Extracting details from: {link}")
    
    if not isinstance(link, str) or not link.startswith('http'):
        print(f"Invalid link: {link}")
        return 'N/A', 'N/A', 'N/A'
    
    driver.get(link)
    time.sleep(1)
    try:
        try:
            current_energy_rating = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//dt[normalize-space(text())="Current Energy Rating"]/following-sibling::dd'))
            ).text
        except Exception as e:
            print(f"Current energy rating not found: {e}")
            current_energy_rating = 'No Data'
        
        try:
            habitable_rooms = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//dt[normalize-space(text())="Habitable Rooms"]/following-sibling::dd'))
            ).text
        except Exception as e:
            print(f"Habitable rooms not found: {e}")
            habitable_rooms = 'No Data'
        print(f"Habitable rooms: {habitable_rooms}")
        
        try:
            year_built_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//dt[normalize-space(text())="Year Built"]/following-sibling::dd'))
            ).text
        except Exception as e:
            print(f"Year built not found: {e}")
            year_built_element = 'No Data'
        print(year_built_element)
        year_built = process_year_built(year_built_element)
        print(f"Year built: {year_built}")
        
        return current_energy_rating, habitable_rooms, year_built
    except Exception as e:
        print(f"Error extracting details from {link}: {e}")
        return 'No Data', 'No Data', 'No Data'

def get_property_details(input_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
    postcodes = df.iloc[:, 2].tolist()
    options = df.iloc[:, 1].tolist()

    links = []
    options_cleaned_list = []
    property_details = []

    # Clean all the commas
    for y in options:
        options_cleaned_list.append(y.replace(",", ""))

    # Initialize the WebDriver 
    driver = webdriver.Chrome()

    try:
        for x in range(len(postcodes)):
            url = 'https://propertychecker.co.uk/results/?postcode='
            driver.get(url + postcodes[x])
            time.sleep(1)  # Wait a bit for the page to load

            try:
                # Wait for the elements to be present
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//strong'))
                )
                address_boxes = driver.find_elements(By.XPATH, '//strong')
                found = False
                print(f"Found {len(address_boxes)} address boxes")
                
                for address_box in address_boxes:
                    address_text = clean_address(address_box.text.replace(",", ""))
                    
                    if compare_addresses(address_text, options_cleaned_list[x]):
                        parent_element = address_box.find_element(By.XPATH, '../../../..')
                        links.append(parent_element.get_attribute('href'))
                        print(f"Found match! {address_text} = {options_cleaned_list[x]}")
                        found = True
                        break
                
                if not found:
                    print(f"No close match found for {postcodes[x]} with option {options[x]}")
                    links.append('N/A')

            except Exception as e:
                print(f"Error locating element for {postcodes[x]} with option {options[x]}: {e}")
                links.append('N/A')

        time.sleep(3)

        # Extract details for each property link
        for link in links:
            if link != 'N/A':
                current_energy_rating, habitable_rooms, year_built = extract_property_details(driver, link)
                property_details.append([link, current_energy_rating, habitable_rooms, year_built])
            else:
                property_details.append(['N/A', 'N/A', 'N/A', 'N/A'])

    finally:
        # Close the WebDriver after scraping details
        driver.quit()

    # Create a DataFrame to save the property details
    property_df = pd.DataFrame(property_details, columns=['Link', 'Current energy ratings', 'Habitable rooms', 'Year built'])
    output_file_bed = '/Users/user/Desktop/property_details.csv'
    # Save the DataFrame to a CSV file
    property_df.to_csv(output_file_bed, index=False)

    print(f"Property details have been saved to {output_file_bed}")
    return property_df

# Example usage
if __name__ == "__main__":
    input_file = '/Users/user//Documents/UCL work/Final Dissertation/Transaction Data/testdata.csv'
    get_property_details(input_file)
