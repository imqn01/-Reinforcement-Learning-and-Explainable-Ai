import imageio.v3 as iio
import numpy as np

def one_hot_decode(label_3D: np.ndarray) -> np.ndarray:
  """
  Converts back from one-hot encoded format (label_3D)
  to original class labels for each pixel (2D)
  """
  return np.argmax(label_3D, axis=-1)

def one_hot_encode(label_2D: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    """
    Converts a segmentation image label array (label_2D) to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label_2D: The 2D array segmentation image label
        label_values: an RGB array of classes (num_classes, 3)

    # Returns
        A 3D array with the same width and height as the input,
        but with a depth size of num_classes (for each pixel)
    """
    semantic_map = []
    for color in label_values:
        equality = np.equal(label_2D, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

def make_2d(map_3d: np.ndarray, palette_array: np.ndarray) -> np.ndarray:
    """
    Converts RGB images to images with number of class as value at [i,j]

    # Arguments
        map_3d: 3D RGB image (H, W, n_ch = 3)
        palette_array: an RGB array of classes (num_classes, 3)

    # Returns
        2D structure where each "pixel" in the image
        has a class number instead of RGB color
    """
    # here we make 3D representation, 3rd dimension has one-hot encoded classes
    replaced_image_onehot = one_hot_encode(map_3d.astype(np.uint8), palette_array) # size (H, W, num_classes)
    # we create 2D representation, where every pixel has the value of its class
    return one_hot_decode(replaced_image_onehot) # size (H, W) - 2D

def make_3d(map_2d: np.ndarray, palette_array: np.ndarray) -> np.ndarray:
    """
    Convert a 2D image where each pixel represents a class label
    into a 3D image where each class label is represented
    by its corresponding RGB color from the palette_array.
    Creates colorized 3D image from the original 2D class map

    # Arguments
        map_2d: array of class labels, size (H, W)
        palette_array: an RGB array of classes (num_classes, 3)

    # Returns
        RGB image (H, W, 3)
    """
    return palette_array[map_2d.astype(np.uint8)]


def read_png_file(image_path: str, printDebug: bool = True) -> np.ndarray:
    original_image_matrix = iio.imread(image_path)

    if original_image_matrix.ndim > 2 and original_image_matrix.shape[-1] > 3:
        if printDebug:
            print(f'image has more than 3 channels, only first 3 channels are used there are {original_image_matrix.shape} channels')
        original_image_matrix = original_image_matrix[:, :, :3]
    return original_image_matrix

def is_valid_idx(image_2D: np.ndarray, selected_idx: int) -> bool:
    if selected_idx[0] < 0 or selected_idx[1] < 0:
        return False
    if selected_idx[0] < image_2D.shape[0]:
        if selected_idx[1] < image_2D.shape[1]:
            return True
    return False

def swap_values(arr: np.ndarray, value1: int, value2: int) -> np.ndarray:
    """
    Swaps occurrences of two specific values (value1 and value2)
    within a 2D NumPy array (arr).

    # Arguments
        arr: A 2D NumPy array where we want to swap the values
        value1: the first value that we want to swap with value2
        value2: the second value that we want to swap with value1

    # Returns
        2D array with swapped values
    """

    # Ensure the values are integers
    value1, value2 = int(value1), int(value2)

    # Step 1: Find the first position of value1 and change it to value2
    pos_value1 = np.argwhere(arr == value1) # returns 2D array
    if pos_value1.size > 0:
        arr[pos_value1[0][0], pos_value1[0][1]] = value2

    # Step 2: Get all positions of value2
    pos_value2 = np.argwhere(arr == value2) # returns 2D array

    # Step 3: Randomly select a position of value2 that is not the original value1's position
    if len(pos_value2) > 1:
        # filter out the position of former value1 (pos_value1) which now also contains value2
        pos_value2 = pos_value2[np.all(pos_value2 != pos_value1[0], axis=1)]
        if pos_value2.size > 0:
            random_pos_value2 = pos_value2[np.random.choice(len(pos_value2))]
            arr[random_pos_value2[0], random_pos_value2[1]] = value1

    return arr