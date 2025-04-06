import pandas as pd

# Read Excel file and convert to CSV
data = pd.read_excel("credit_data.xls", header=1)  # Skip the first row (title)
data.to_csv("credit_data.csv", index=False)
print("Dataset converted to CSV!")
