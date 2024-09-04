import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import random
import re

# Load the CSV file
csv_file_path = '/Users/user/Documents/UCL work/Final Dissertation/District Data/unique_not_ok.csv'
data = pd.read_csv(csv_file_path)

# Access the column with the postcode data and convert it to a list
postcodelist = data.iloc[:, 0].tolist()

# Initialize the Chrome driver using WebDriver Manager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Log in
collected_data = []
driver.get('https://propertydata.co.uk/local-data')
Login_button = driver.find_element(By.XPATH, "//a[contains(@class, 'btn btn-tertiary-orange-outline block px-3 mb-3 lg:mb-0 xl:px-6') and contains(text(), 'Log in')]")
Login_button.click()
Email_Input = driver.find_element(By.CSS_SELECTOR, "input[name='email']")
Email_Input.send_keys('XXX')
Password_Input = driver.find_element(By.CSS_SELECTOR, "input[name='password']")
Password_Input.send_keys('XXX')
time.sleep(random.uniform(1, 3))
Submit_button = driver.find_element(By.CSS_SELECTOR, "input[value='Login']")
Submit_button.click()
time.sleep(random.uniform(3, 5))
wait = WebDriverWait(driver, 10)

def save_intermediate_data(data, path):
    """Save collected data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Data has been collected and saved to {path}")

try:
    for index, postcode in enumerate(postcodelist):
        try:
            # Open the website
            driver.get('https://propertydata.co.uk/local-data')
            time.sleep(random.uniform(2, 4))
            # Find the search input element
            search_box = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
            
            # Clear the search box, enter the postcode, and submit the form
            search_box.clear()
            search_box.send_keys(postcode)
            
            # Explicit wait to ensure the search button is clickable
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[value='Search']"))
            )
            
            # Click the search button using JavaScript to avoid interception
            driver.execute_script("arguments[0].click();", search_button)
            time.sleep(random.uniform(2, 4))
            # Wait for a few seconds to let the page load the results
            generate_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[value='Generate data']"))
            )
            generate_button.click()
            
            # Handle the pop-up if it appears
            try:
                run_search_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Run search')]"))
                )
                run_search_button.click()
                time.sleep(2)
            except:
                # Pop-up did not appear, continue as usual
                pass
            
            try:
                sold_prices_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'btn border-0 rounded-none px-4 flex-shrink-0') and contains(text(), 'Sold prices')]"))
                )
                sold_prices_button.click()
                
                data_view_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//img[contains(@src, 'images/v2/icons/table-view-light.svg')]/parent::a"))
                )
                data_view_button.click()
            
                download_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'download btn btn-tertiary-orange block flex-shrink-0 mb-2')]"))
                )
                download_link = download_button.get_attribute('href')
                download_link_temp = re.sub(r'download=1', r'download=1&max_age=96', download_link)          
                driver.get(download_link_temp)
            
                # Optionally, save or process the search results here
                print(f"Data saved for {postcode}")
            
            except TimeoutException as e:
                print(f"Could not find or click the required button for {postcode}: {e}")
            
            except Exception as e:
                print(f"An error occurred for {postcode}: {e}")
            
            # District Data Collection
            # try:
            #     Population_element = driver.find_element(By.XPATH, "//span[contains(text(), 'Population:')]/following-sibling::span")
            #     Population_data = Population_element.text

            #     Households_element = driver.find_element(By.XPATH, "//span[contains(text(), 'Households:')]/following-sibling::span")
            #     Households_data = Households_element.text

            #     Density_element = driver.find_element(By.XPATH, "//span[contains(text(), 'Density:')]/following-sibling::span")
            #     Density_data = Density_element.text.strip('/mi²')
            # except:
            #     Population_data = Households_data = Density_data = None

            # try:
            #     # Demographics section
            #     Demographic_button = wait.until(
            #         EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'no-underline hover:opacity-70') and contains(text(), 'Demographics')]"))
            #     )
            #     Demographic_button.click()
                
            #     work_button = wait.until(
            #         EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'btn border-0 rounded-none px-4 border-primary-white border-opacity-50 border-r ') and contains(text(), 'Work')]"))
            #     )
            #     work_button.click()
                
            #     average_income_element = wait.until(
            #         EC.presence_of_element_located((By.XPATH, "//p[contains(text(), 'average household income')]/preceding-sibling::p"))
            #     )
            #     average_income_data = average_income_element.text.strip('£')
            
            # except TimeoutException as e:
            #     average_income_data = None
            #     print(f"TimeoutException occurred: {e}")
            
            # except Exception as e:
            #     average_income_data = None
            #     print(f"An error occurred: {e}")
            
            # try:
            #     # Area section
            #     Area_button = wait.until(
            #         EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'no-underline hover:opacity-70') and contains(text(), 'Area')]"))
            #     )
            #     Area_button.click()
                
            #     School_button = wait.until(
            #         EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'btn border-0 rounded-none px-4 border-primary-white border-opacity-50 border-r ') and contains(text(), 'Schools')]"))
            #     )
            #     School_button.click()
                
            #     State_school_element = wait.until(
            #         EC.presence_of_element_located((By.XPATH, "(//span[contains(text(), 'state schools and')]/preceding-sibling::span)[1]"))
            #     )
            #     State_school = State_school_element.text
            
            # except TimeoutException as e:
            #     State_school = None
            #     print(f"TimeoutException occurred: {e}")
            
            # except Exception as e:
            #     State_school = None
            #     print(f"An error occurred: {e}")

            # # Append the data to the list
            # collected_data.append({
            #     'Postcode': postcode,
            #     'Population': Population_data,
            #     'Households': Households_data,
            #     'Density': Density_data,
            #     'Five Year Growth': 'N/A',
            #     'Average Income': average_income_data,
            #     'Crime Rate': 'N/A',
            #     'State Schools': State_school,
            #     'Green Space': 'N/A'
            # })

            # # Save intermediate data every 1 postcodes
            # if (index + 1) % 1 == 0:
            #     save_intermediate_data(collected_data, '/Users/user/Documents/UCL work/Final Dissertation/collected_data_intermediate.csv')

        except TimeoutException as e:
            print(f"Timeout or other error occurred, restarting loop for {postcode}: {e}")
            continue  # Restart the current loop iteration

        time.sleep(random.uniform(1, 3))  # Random wait before next iteration

finally:
    # Ensure the browser is closed properly
    driver.quit()

# Convert the collected data to a DataFrame and save it as a CSV file
# save_intermediate_data(collected_data, '/Users/user/Documents/UCL work/Final Dissertation/collected_data_final.csv')
