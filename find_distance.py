import pandas as pd
import numpy as np

def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dLat = lat2_rad - lat1_rad
    dLon = lon2_rad - lon1_rad
    
    a = np.sin(dLat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dLon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    R = 6371.0
    
    return R * c

df = pd.read_csv('cleaned_data_no_outliers.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

start_time = pd.to_datetime('2023-03-01', utc=True)
end_time = pd.to_datetime('2023-03-24', utc=True)

df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

df_sampled = df_filtered.sample(n=100, random_state=42) # sample of 100

proximity_events = []

lat_lon_pairs = df_sampled[['lat', 'lon']].to_numpy()
mmsi_pairs = df_sampled[['mmsi']].to_numpy()
timestamps = df_sampled[['timestamp']].to_numpy()

for i in range(len(lat_lon_pairs)):
    for j in range(i + 1, len(lat_lon_pairs)):
        lat1, lon1 = lat_lon_pairs[i]
        lat2, lon2 = lat_lon_pairs[j]
        mmsi1, mmsi2 = mmsi_pairs[i][0], mmsi_pairs[j][0]
        timestamp1, timestamp2 = timestamps[i][0], timestamps[j][0]
        
        if mmsi1 != mmsi2:
            distance = haversine_vectorized(lat1, lon1, lat2, lon2)
            threshold_distance = 10.0
            
            if distance <= threshold_distance:
                proximity_events.append({
                    'mmsi': mmsi1,
                    'other_vessel_mmsi': mmsi2,
                    'timestamp': timestamp1
                })

df_proximity_events = pd.DataFrame(proximity_events)

df_proximity_events['vessel_proximity'] = df_proximity_events.groupby(['mmsi', 'timestamp'])['other_vessel_mmsi'] \
    .transform(lambda x: ','.join(map(str, sorted(set(x)))))  
    
df_proximity_events = df_proximity_events.drop_duplicates(subset=['mmsi', 'timestamp'])

df_proximity_events = df_proximity_events.rename(columns={'vessel_proximity': 'vessel_proximity'})

# Group by mmsi and timestamp
df_grouped = df_proximity_events.groupby(['mmsi', 'timestamp'])['vessel_proximity'].first().reset_index()

df_grouped.to_csv('vessel_proximity.csv', index=False)

print("Save Proximity events into to 'vessel_proximity.csv'.")
