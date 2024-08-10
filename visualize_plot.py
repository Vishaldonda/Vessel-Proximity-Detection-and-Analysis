import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
df = pd.read_csv('vessel_proximity.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Initialize a plot
plt.figure(figsize=(12, 8))

# Iterate through unique timestamps
for timestamp in df['timestamp'].unique():
    # Filter data for the current timestamp
    df_filtered = df[df['timestamp'] == timestamp]
    
    # Plot each MMSI against its vessel proximities
    for _, row in df_filtered.iterrows():
        plt.scatter(row['mmsi'], row['vessel_proximity'], label=f"{timestamp}", alpha=0.7)

# Enhance plot
plt.title('MMSI vs Vessel Proximities at Each Timestamp')
plt.xlabel('MMSI')
plt.ylabel('Vessel Proximity')
plt.xticks(rotation=90)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Show the plot
plt.show()
