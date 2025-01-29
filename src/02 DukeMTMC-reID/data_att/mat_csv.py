import scipy.io
import pandas as pd
import os
import numpy as np

# Function to load the .mat file with error handling
def load_mat_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    try:
        data = scipy.io.loadmat(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading .mat file: {e}")

# Function to recursively explore the contents of a numpy structured array
def explore_struct(struct, indent=0):
    if isinstance(struct, np.ndarray) and struct.dtype.names is not None:
        for field in struct.dtype.names:
            value = struct[field]
            if isinstance(value, np.ndarray) and value.dtype.names is not None:
                print(" " * indent + f"Field: {field}, Type: {type(value)}, Shape: {value.shape}")
                explore_struct(value, indent + 4)
            else:
                print(" " * indent + f"Field: {field}, Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    elif isinstance(struct, np.ndarray):
        print(" " * indent + f"Array Shape: {struct.shape}, Type: {struct.dtype}")
    else:
        print(" " * indent + f"Value: {struct}")

# Function to convert MATLAB struct to pandas DataFrame
def struct_to_dataframe(struct, attribute_names, image_paths):
    data_dict = {}
    for name in attribute_names:
        data = struct[name].flatten()
        if len(data) == 1 and isinstance(data[0], np.ndarray):
            # Unpack nested array if it contains an ndarray with more elements
            data = data[0].flatten()
        data_dict[name] = data
    df = pd.DataFrame(data_dict)
    # Match the length of image paths with the length of data
    min_length = min(len(image_paths), len(df))
    df = df.iloc[:min_length]
    image_paths = image_paths[:min_length]
    df.insert(0, 'path', image_paths)
    
    # Extract person ID from image names and add as a column
    person_ids = [img_path.split('_')[0] for img_path in image_paths]  # Extract person ID from the beginning of the image name
    df.insert(1, 'person_id', person_ids)

    # Correct the image_index to ensure it matches the person_id consistently
    df['image_index'] = df['person_id'].apply(lambda x: f'{int(x):04d}')

    return df

# Get image file paths from the directories
def get_image_paths(directory):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    return image_files

# Load the .mat file
data_path = '../Data/DukeMTMC-attribute/duke_attribute.mat'
data = load_mat_file(data_path)

# Display contents of the .mat file
print("Contents of duke_attribute.mat:")
for key in sorted(data):  # Sort keys alphabetically
    if key.startswith('__'):
        continue  # Skip meta keys
    print(f"Key: {key}, Type: {type(data[key])}, Shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")

# Access and explore 'duke_attribute' struct if it exists
if 'duke_attribute' in data:
    duke_attribute = data['duke_attribute']
    print("\nDetails of 'duke_attribute':")
    explore_struct(duke_attribute)
    
    # Extract train and test data structs
    train_data_struct = duke_attribute['train'][0, 0]
    test_data_struct = duke_attribute['test'][0, 0]

    # Get attribute names
    attribute_names = train_data_struct.dtype.names

    # Define paths to image directories
    train_images_path = '../Data/archive/bounding_box_train'
    test_images_path = '../Data/archive/bounding_box_test'

    # Get image paths
    train_image_paths = get_image_paths(train_images_path)
    test_image_paths = get_image_paths(test_images_path)

    # Convert structs to DataFrames
    print("Converting train data struct to DataFrame...")
    train_df = struct_to_dataframe(train_data_struct, attribute_names, train_image_paths)
    print("Train data successfully converted.")

    print("Converting test data struct to DataFrame...")
    test_df = struct_to_dataframe(test_data_struct, attribute_names, test_image_paths)
    print("Test data successfully converted.")

    # Correct the person_id column to ensure it matches the image path identifier correctly
    train_df['person_id'] = train_df['path'].apply(lambda x: x.split('_')[0])
    test_df['person_id'] = test_df['path'].apply(lambda x: x.split('_')[0])

    # Convert person_id to integer to ensure consistency
    train_df['person_id'] = train_df['person_id'].astype(int)
    test_df['person_id'] = test_df['person_id'].astype(int)

    # Correct image_index to match person_id for all images consistently
    train_df['image_index'] = train_df['person_id'].apply(lambda x: f'{x:04d}')
    test_df['image_index'] = test_df['person_id'].apply(lambda x: f'{x:04d}')

    # Save DataFrames to CSV files
    train_csv_path = 'duke_attribute_train.csv'
    test_csv_path = 'duke_attribute_test.csv'

    print(f"Saving train DataFrame to {train_csv_path}...")
    train_df.to_csv(train_csv_path, index=False)
    print("Train CSV file has been successfully created.")

    print(f"Saving test DataFrame to {test_csv_path}...")
    test_df.to_csv(test_csv_path, index=False)
    print("Test CSV file has been successfully created.")
else:
    print("The key 'duke_attribute' was not found in the loaded .mat file.")