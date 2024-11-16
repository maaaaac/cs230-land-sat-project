"""
A random record is selected from the dataset, and a google street view image is returned
at the location of record's coordinates.

Keyword arguments:
    csv_file_path - path to dataset with manhole records

Return: print and saves the google street view image of the coordinates provided
"""
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import random
from pyproj import Transformer
import os

def get_api_env_var():
    # Get the API key from environment variable
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    return api_key

def get_xy_record_values(csv_file_path, n, selected_ids_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Drop records with missing coordinates
    df = df.dropna(subset=['X', 'Y', 'OBJECTID'])
    if df.empty:
        raise ValueError("No valid coordinates found in the CSV file.")
    
    # Check if selected_ids_path exists and load it
    if os.path.exists(selected_ids_path) and os.stat(selected_ids_path).st_size > 0:
        selected_ids = pd.read_csv(selected_ids_path)['OBJECTID'].tolist()
    else:
        selected_ids = []

    # If a list of already selected IDs is provided, exclude them
    if selected_ids is not None:
        df = df[~df['OBJECTID'].isin(selected_ids)]
        if df.empty:
            raise ValueError("No records left to select after filtering out previously selected IDs.")

    # Select a random record
    record = df.sample(n=n, random_state=42)
    
    # Extract X and Y coordinates
    x, y, ObjectID = record['X'], record['Y'], record['OBJECTID']

    # Add the selected AssetIDs to the selected_ids list
    new_selected_ids = ObjectID.tolist()
    selected_ids.extend(new_selected_ids)
    
    # Save the updated selected IDs back to the CSV file
    pd.DataFrame({'OBJECTID': selected_ids}).to_csv(selected_ids_path, index=False)
    

    return x, y, ObjectID

def convert_coordinates(x, y):
    # Ensure coordinates are numeric
    x = [float(val) for val in x]
    y = [float(val) for val in y]
    
    # Convert coordinates to latitude and longitude
    try:
        transformer = Transformer.from_crs("EPSG:26986", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
    except Exception as e:
        raise ValueError(f"Error converting coordinates: {e}")
    
    return lon, lat

def fetch_street_view_images_for_directions(csv_file_path, output_image_path, n, selected_ids_path):
    # Get the API key
    api_key = get_api_env_var()

    # Pick random record and retrieve x, y values
    x, y, ObjectID = get_xy_record_values(csv_file_path, n, selected_ids_path)
    
    # Convert coordinates to latitude and longitude
    lon, lat = convert_coordinates(x, y)
    
    # Define base URL for Google Street View API
    base_url = 'https://maps.googleapis.com/maps/api/streetview'
    
    # Define three directions to cover 360 degrees with fov=120
    directions = {
        'N': 0,
        'SE': 120,
        'SW': 240
    }

    for i in range(n):
    
        # Fetch and save images for each direction
        for direction, heading in directions.items():
            params = {
                'size': '600x300',
                'location': f'{lat[i]},{lon[i]}',
                'fov': 120,
                'heading': heading,
                'pitch': -60,
                'key': api_key
            }
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                save_as = f'{output_image_path}/{ObjectID.iloc[i]}_{direction}.jpg'  # Use .iloc for ObjectID
                image.save(save_as)
                #print(f"Image saved as {save_as} for {direction}")
            except requests.RequestException as e:
                print(f"Error fetching {direction} image: {e}")

# Example usage
csv_file_path = '/home/ec2-user/cs230/data/mass_manhole.csv'
#selected_ids_path = 'AdditionalData/selected_ids.csv'
selected_ids_path = 'predict/FetchedImages/selected_ids.csv'
output_image_path = 'predict/FetchedImages'
fetch_street_view_images_for_directions(csv_file_path, output_image_path, n=10, selected_ids_path=selected_ids_path)
