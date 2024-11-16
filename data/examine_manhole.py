import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import argparse
from pyproj import Transformer

def get_api_env_var():
    # Get the API key from environment variable
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    return api_key

def get_xy_record_values(csv_file_path, object_id):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Drop records with missing coordinates
    df = df.dropna(subset=['X', 'Y', 'OBJECTID'])
    if df.empty:
        raise ValueError("No valid coordinates found in the CSV file.")
    
    # Filter the dataframe to get the record with the specified OBJECTID
    record = df[df['OBJECTID'] == object_id]
    if record.empty:
        raise ValueError(f"OBJECTID {object_id} not found in the CSV file.")
    
    # Print the ManholeType for the specified OBJECTID
    manhole_type = record['ManholeType'].values[0]
    print(f"ManholeType for OBJECTID {object_id}: {manhole_type}")

    # Extract X and Y coordinates
    x, y = record['X'].values[0], record['Y'].values[0]

    return x, y, object_id

def convert_coordinates(x, y):
    # Convert coordinates to latitude and longitude
    try:
        transformer = Transformer.from_crs("EPSG:26986", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
    except Exception as e:
        raise ValueError(f"Error converting coordinates: {e}")
    
    return lon, lat

def fetch_street_view_images_for_directions(csv_file_path, object_id):
    # Get the API key
    api_key = get_api_env_var()

    # Retrieve x, y values based on the specified OBJECTID
    x, y, object_id = get_xy_record_values(csv_file_path, object_id)
    
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

    # List to store the paths of saved images
    saved_images = []

    # Fetch and save images for each direction
    for direction, heading in directions.items():
        params = {
            'size': '600x300',
            'location': f'{lat},{lon}',
            'fov': 120,
            'heading': heading,
            'pitch': -60,
            'key': api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            save_as = f'/home/ec2-user/cs230/data/images_to_examine/{object_id}_{direction}.jpg'
            image.save(save_as)
            saved_images.append(save_as)
            print(f"Image saved as {save_as} for {direction}")
        except requests.RequestException as e:
            print(f"Error fetching {direction} image: {e}")

    # Ask for user confirmation before deleting images
    delete_confirmation = input("Do you want to delete the saved images? (y/n): ").strip().lower()
    if delete_confirmation == 'y':
        for image_path in saved_images:
            try:
                os.remove(image_path)
                print(f"Deleted image: {image_path}")
            except FileNotFoundError:
                print(f"File not found, could not delete: {image_path}")
    else:
        print("Images were not deleted.")

if __name__ == "__main__":
    # Argument parser for handling command-line arguments
    parser = argparse.ArgumentParser(description="Fetch street view images for a given OBJECTID.")
    parser.add_argument("--object_id", type=int, required=True, help="Specify the OBJECTID to fetch data for.")
    args = parser.parse_args()

    # Example CSV file path
    csv_file_path = '/home/ec2-user/cs230/data/mass_manhole.csv'

    # Run the main function
    fetch_street_view_images_for_directions(csv_file_path, object_id=args.object_id)
