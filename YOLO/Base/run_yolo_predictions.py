import subprocess
import os
import random
from pathlib import Path
from train.GoogleStreetView import fetch_street_view_images_for_directions

def run_yolo_predictions_on_random_images(coord_directory, weights_path, output_directory, num_images=5, confidence_threshold=0.25):
    """
    Runs YOLO predictions on a random subset of images from a directory.

    Args:
        coord_directory (str): Path to the directory containing coordinates.
        weights_path (str): Path to the YOLO model weights file.
        output_directory (str): Directory to save YOLO output images.
        num_images (int): Number of random images to predict.
        confidence_threshold (float): Confidence threshold for predictions.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    csv_file_path = '/home/ec2-user/cs230/data/mass_manhole.csv'
    selected_ids_path = 'predict/FetchedImages/selected_ids.csv'
    output_image_path = 'predict/FetchedImages'
    fetch_street_view_images_for_directions(csv_file_path, output_image_path, n=1, selected_ids_path=selected_ids_path)


    # Get all image paths from the directory
    image_paths = list(Path(image_directory).glob("*.jpg"))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {image_directory}")

    # Select random images
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))

    for image_path in selected_images:
        # Construct the command for running detect.py on each image
        command = [
            "python", "yolov5/detect.py",
            "--weights", weights_path,
            "--source", str(image_path),
            "--img", "640",  # Adjust image size if needed
            "--conf", str(confidence_threshold),
            "--project", output_directory,
            "--name", "predictions",  # Subdirectory for predictions
            "--exist-ok",  # Avoid creating a new folder for each prediction
        ]

        try:
            # Execute the YOLO detection command
            subprocess.run(command, check=True)
            print(f"Prediction completed for {image_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running YOLO prediction on {image_path}: {e}")

if __name__ == "__main__":
    # Define paths
    image_directory = "AdditionalData/2"  # Path to saved images
    weights_path = "yolov5/runs/train/exp7/weights/best.pt"  # Path to your trained weights
    output_directory = "AdditionalData/2/yolo_predictions"  # Directory to save predictions
    num_images_to_predict = 10  # Number of random images to predict

    # Run YOLO predictions on random images
    run_yolo_predictions_on_random_images(image_directory, weights_path, output_directory, num_images=num_images_to_predict)
