import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import arff
from skimage.color import rgb2luv
import scipy as sc
import warnings
import torch
from skimage import color

# PARTIALLY TAKEN FROM: https://www.kaggle.com/code/xuandat7/multilabel-multiclass-image-classification

warnings.filterwarnings("ignore")

# Function to compute spatial color moments for a single image
def compute_spatial_color_moments(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Convert from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV) / 255.0


    # Convert RGB to Luv using skimage's rgb2luv
    # luv_image = rgb2luv(image / 255.0)  # Normalizing the image to range [0, 1]

    # Define grid size (7x7)
    grid_size = (7, 7)
    height, width, _ = luv_image.shape
    grid_height = height // grid_size[0]
    grid_width = width // grid_size[1]

    # Feature vector to store the results
    feature_vector = [[[], []] for i in range(3)]
    # Loop through each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract the block
            block = luv_image[i * grid_height:(i + 1) * grid_height, j * grid_width:(j + 1) * grid_width, :]


            # Compute first (mean) and second (variance) moments for each channel
            for channel in range(3):  # L, u, v channels
                mean = np.mean(block[:, :, channel])
                std = np.std(block[:, :, channel])

                # Append to feature vector
                feature_vector[channel][0].append(mean)
                feature_vector[channel][1].append(std)
                # feature_vector.extend([mean, std])

    feature_vector = np.asarray(feature_vector).reshape(1, 294)

    # VISUALIZATION OF CONVERSION
    # extract means (first moments) for L, u, v channels
    luv_means = feature_vector.reshape(3, 2, 7 * 7)[:, 0].transpose()
    # Undo standard scaling and return to rgb
    rgb_means = color.luv2rgb(luv_means * np.array([[100, 354, 262]]) + np.array([[0, -134, -140]]))

    # reshape into 7x7 grid for each Luv channel

    # Convert Luv to RGB
    rgb_image = rgb_means.reshape(7,7,3)
    # Step 3: Visualize the Reconstructed Low-Resolution Image
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_image)
    plt.title("Reconstructed Low-Resolution Image all RGB channels")
    plt.axis("off")
    plt.show()

    cv2.imshow('image', image[...,::-1])
    cv2.waitKey(0)

    return np.array(feature_vector)

# Function to process the dataset and return tensors
def process_dataset(image_folder, df_train):
    all_features = []

    for idx, row in df_train.iterrows():
        image_path = os.path.join(image_folder, row['image'])
        try:
            features = compute_spatial_color_moments(image_path)
            all_features.append(features)
        except FileNotFoundError as e:
            print(e)

    # Convert to tensor
    feature_tensor = torch.tensor(all_features, dtype=torch.float32)
    target_tensor = torch.tensor(df_train['labels'].tolist(), dtype=torch.float32)

    return feature_tensor, target_tensor


def save_to_arff(features, targets, output_path, class_list):
    attr_columns = [f"attr{i+1}" for i in range(features.shape[2])]
    class_columns = class_list

    # Create DataFrame for ARFF
    data = pd.DataFrame(features.numpy().reshape(2000, 294), columns=attr_columns)
    for i, cls in enumerate(class_columns):
        data[cls] = targets[:, i].numpy().astype(int).astype(str)

    # Convert to ARFF format
    arff_data = {
        'description': '',
        'relation': 'scene_data',
        'attributes': [(col, 'REAL') for col in attr_columns] + [(cls, ['0', '1']) for cls in class_columns],
        'data': data.values.tolist()
    }

    # Save to file
    with open(output_path, 'w', encoding="utf8") as f:
        arff.dump(arff_data, f)


# Path setup
base_dir = "./miml-image-data"
image_folder = os.path.join(base_dir, "original")  # Adjust if your images are stored elsewhere

# File containing target labels
mat_file_path = os.path.join(base_dir, 'processed', 'miml data.mat')
mat_file = sc.io.loadmat(mat_file_path)

# Array of target labels
target_array = mat_file['targets'].T
target_list = [[j if j == 1 else 0 for j in row] for row in target_array]

# Class list
class_list = [j[0][0] for j in mat_file['class_name']]

print('Class labels:', class_list)

# Create DataFrame for images and labels
file_list = [str(a) + '.jpg' for a in range(1, 2001, 1)]
df_train = pd.DataFrame({'image': file_list, 'labels': target_list})

# One-hot to categorical
def target_label(x):
    categorical_list = list(np.array(class_list)[np.nonzero(x)])
    return categorical_list

df_train['txt_labels'] = df_train['labels'].apply(lambda x: target_label(x))

print('Training data example:', df_train.head())

# Process dataset
df_train['txt_labels'] = df_train['labels'].apply(lambda x: target_label(x))
all_features, all_targets = process_dataset(image_folder, df_train)

print("Features shape:", all_features.shape)
print("Targets shape:", all_targets.shape)

# Save processed dataset to ARFF format
def pandas_to_arff(df, filename, relation_name="relation"):
    """
    Convert a Pandas DataFrame to an ARFF file.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert
    filename : str
        The name of the output ARFF file
    relation_name : str, optional
        The name of the relation in the ARFF file (default: "relation")
    """

    def get_arff_type(dtype):
        """Helper function to convert pandas dtypes to ARFF types."""
        if 'int' in str(dtype):
            return 'NUMERIC'
        elif 'float' in str(dtype):
            return 'NUMERIC'
        elif 'bool' in str(dtype):
            return '{True, False}'
        elif 'category' in str(dtype):
            categories = dtype.categories.tolist()
            return '{' + ','.join(map(str, categories)) + '}'
        else:
            return 'STRING'

    # Open the output file
    with open(filename, 'w') as f:
        # Write the relation name
        f.write(f'@RELATION {relation_name}\n\n')

        # Write attributes
        for column in df.columns:
            attr_type = get_arff_type(df[column].dtype)
            f.write(f'@ATTRIBUTE "{column}" {attr_type}\n')

        # Write data
        f.write('\n@DATA\n')

        # Convert DataFrame to CSV format without header and index
        data_str = df.to_csv(index=False, header=False, line_terminator='\n')
        f.write(data_str)

#
# def save_to_arff(features, targets, output_path, class_list):
#     attr_columns = [f"attr{i+1}" for i in range(features.shape[2])]
#     class_columns = class_list
#
#     # Create DataFrame for ARFF
#     data = pd.DataFrame(features.numpy().reshape(2000, 294), columns=attr_columns)
#     for i, cls in enumerate(class_columns):
#         data[cls] = targets[:, i].numpy()
#
#     pandas_to_arff(data, output_path, "scene dataset")

output_path = os.path.join(base_dir, "scene_data_manual.arff")
save_to_arff(all_features, all_targets, output_path, class_list)

print(f"Dataset saved to {output_path}")
