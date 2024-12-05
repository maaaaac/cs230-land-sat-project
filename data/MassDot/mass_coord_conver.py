import pandas as pd
from pyproj import Transformer

# Initialize transformer from EPSG:26986 to EPSG:4326 (WGS84)
transformer = Transformer.from_crs(26986, 4326)

# Read the CSV file
df = pd.read_csv('/home/ec2-user/cs230/data/mass_manhole.csv', low_memory=False)  # Replace with your input file name

# Define a function to convert X, Y to latitude and longitude
def convert_coordinates(x, y):
    lon, lat = transformer.transform(x, y)
    return pd.Series([lat, lon])

# Apply the conversion to each row in the dataframe
df[['Latitude', 'Longitude']] = df.apply(lambda row: convert_coordinates(row['X'], row['Y']), axis=1)

# Save the updated dataframe with Latitude and Longitude to a new CSV file
df.to_csv('mass_manhole_conv_coord.csv', index=False)  # Replace with your desired output file name

print("Conversion complete. The new CSV file has been saved.")
