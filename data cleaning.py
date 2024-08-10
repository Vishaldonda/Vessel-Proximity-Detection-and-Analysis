import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('sample_data.csv')

# 1. Remove duplicates
duplicates = df[df.duplicated()]
print(f"Number of duplicate rows present: {len(duplicates)}")
df_cleaned = df.drop_duplicates() # remove duplicates
print("Removed duplicate rows")

# 2. Check for missing values
missing_values = df_cleaned.isnull().sum()
if missing_values.any():
    print("Missing values in each column:")
    print(missing_values)
else:
    print("No missing values in the DataFrame.")

# 3. Check for outliers using Z-score
z_scores = np.abs(stats.zscore(df_cleaned[['lat', 'lon']]))
outliers_present = (z_scores > 3).any()
if outliers_present.any():
    print("\nOutliers are present in the data.")
    non_outliers_mask = (z_scores <= 3).all(axis=1)
    df_no_outliers = df_cleaned[non_outliers_mask]
    
    df_no_outliers.to_csv('cleaned_data_no_outliers.csv', index=False)
    print(f"Number of outliers removed: {len(df_cleaned) - len(df_no_outliers)}")
else:
    print("\nNo outliers detected in the data.")
    df_cleaned.to_csv('remove_duplicates.csv',index=False)