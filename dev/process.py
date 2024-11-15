!gdown https://drive.google.com/uc?id=1zpT6gHzvbr21_Vq4NjpRY0JGSaQOSANa
!unzip fifa.zip

import os
from PIL import Image

# Directory containing images
directory = "/content/Images"  # Replace with your directory path

# Initialize a set to store unique image formats and a list to store invalid files
image_formats = set()
invalid_files = []

# Loop through files and try to open as images
for file in os.listdir(directory):
    file_path = os.path.join(directory, file)
    if os.path.isfile(file_path):
        try:
            with Image.open(file_path) as img:
                image_formats.add(img.format)  # Get the image format
        except Exception as e:
            # Add file to the list of invalid files
            invalid_files.append(file)
            print(f"File {file} is not a valid image or could not be opened: {e}")

# Display results
print("Image types based on actual format:", image_formats)
print(f"Number of invalid files: {len(invalid_files)}")
print("List of invalid files:", invalid_files)

import pandas as pd
data = pd.read_csv("data.csv")
data['idnumber'] = data.index
data.to_csv("datafix.csv", index=False)

#4
import pandas as pd

# Sample list of filenames
file_list = invalid_files

# Extract the numeric part of the filenames (without extension)
file_numbers = [int(file.split('.')[0]) for file in file_list]

# Load the CSV file
csv_path = "datafix.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_path)

# Remove rows where 'idnumber' matches any of the numbers in the file list
filtered_data = data[~data['idnumber'].isin(file_numbers)]

# Optionally, save the filtered DataFrame back to a CSV
filtered_csv_path = "filtered_data_fix.csv"
filtered_data.to_csv(filtered_csv_path, index=False)

# Print or return the filtered DataFrame
print(f"Filtered data saved to {filtered_csv_path}")

import pandas as pd
data = pd.read_csv("filtered_data_fix.csv")
data['realidnumber'] = data.index
data.to_csv("filtered_data_fix2.csv", index=False)

#6
import os
import pandas as pd

# Directory containing images
image_directory = "/content/Images"  # Replace with your directory path

# Load your CSV (make sure it contains an 'idnumber' column)
csv_path = "/content/filtered_data_fix2.csv"  # Replace with your CSV path
data = pd.read_csv(csv_path)

# Get a list of image filenames in the directory
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

# Create a new column 'image_path' initialized to None
data['image_path'] = None

# Loop through image files and match based on 'idnumber'
for image_file in image_files:
    # Extract the numeric part of the filename (e.g., '4563.png' -> 4563)
    image_id = int(image_file.split('.')[0])

    # Check if the 'idnumber' exists in the DataFrame and update the 'image_path'
    if image_id in data['idnumber'].values:
        # Find the row with the matching 'idnumber' and set the 'image_path'
        data.loc[data['idnumber'] == image_id, 'image_path'] = os.path.join(image_directory, image_file)

# Save the updated DataFrame with image paths to a new CSV
updated_csv_path = "/content/filtered_data_fix2_withimages.csv"  # Replace with your desired path
data.to_csv(updated_csv_path, index=False)

print(f"CSV with image paths saved to {updated_csv_path}")




import pandas as pd

# Load the CSV file
df = pd.read_csv("/content/filtered_data_fix2_withimages.csv")  # Replace with the path to your CSV file

# Replace '/content/Images/' with '/workspaces/blank-app/extracted_files/Images' in the 'image_path' column
df['image_path'] = df['image_path'].str.replace('/content/Images/', 'data/Images/', regex=False)

# Save the modified DataFrame to a new CSV
df.to_csv("/content/filtered_data_fix2_withimages2.csv", index=False)  # Replace with the desired output file name