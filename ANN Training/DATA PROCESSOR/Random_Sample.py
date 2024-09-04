import pandas as pd

def random_sample(file_path, output_path, n, random_state=42):
    data = pd.read_csv(file_path)
    sampled_data = data.sample(n=n, random_state=random_state)
    sampled_data.to_csv(output_path, index=False)
    print(f"Sampled data exported to {output_path}")

# Commented out example usage

if __name__ == "__main__":
    print('not my business')
    file_path = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/merged.csv'  # Replace with the path to your CSV file
    output_path ='/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/sampled.csv'
    n = 10000
    random_sample(file_path, output_path, n, random_state=42)

# file_path = '/path/to/input.csv'
# output_path = '/path/to/output.csv'
# random_sample(file_path, output_path)
