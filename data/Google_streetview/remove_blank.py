import os

# Define the paths to each set
sets = ['train', 'valid', 'test']

base_dir_path = '/home/ec2-user/cs230/data/custom_google_data/Custom MassDOT Dataset.v1i.yolov5pytorch'

sets = [os.path.join(base_dir_path, split) for split in sets]
# Function to count the number of images in the images folder
def count_images(images_folder):
    return len([img for img in os.listdir(images_folder) if img.endswith('.jpg')])

# Loop over each set
for set_type in sets:
    print(f'\nRemoving non-labeled images in {set_type.split} set')
    images_folder = os.path.join(set_type, 'images')
    labels_folder = os.path.join(set_type, 'labels')
    
    # Count images before cleanup
    initial_count = count_images(images_folder)
    

    # Iterate through each image file in the images folder
    for image_file in os.listdir(images_folder):
        if image_file.endswith('.jpg'):
            # Generate corresponding label file path
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_folder, label_file)

            # Check if the label file exists and is empty
            if os.path.isfile(label_path) and os.path.getsize(label_path) == 0:
                # Label file is empty, so delete both the image and label
                image_path = os.path.join(images_folder, image_file)
                os.remove(image_path)
                os.remove(label_path)
                print(f"Removed {image_path.split('/')[-1]} and {label_path.split('/')[-1]}")

    # Count images after cleanup
    final_count = count_images(images_folder)
    print(f"Initial count of images in '{set_type}': {initial_count}")
    print(f"Final count of images in '{set_type}': {final_count}\n\n\n")