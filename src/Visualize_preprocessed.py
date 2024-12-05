import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_original_images(data_dir):
    original_images = []
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        images_path = os.path.join(label_path, 'images')
        if not os.path.exists(images_path):
            continue

        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for image_name in image_files:
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                original_images.append(image)
    return original_images

def visualize_preprocessed(data_directory, processed_image_file, mask_file):
    # Load original images directly from the dataset
    original_images = load_original_images(data_directory)
    # Load processed images and masks from .npy files
    processed_images = np.load(processed_image_file, allow_pickle=True)
    masks = np.load(mask_file, allow_pickle=True)

    # Ensure we have enough images for visualization
    num_samples = min(5, len(original_images), len(processed_images))

    # Visualize the samples
    for i in range(num_samples):  # Display up to 3 samples
        plt.figure(figsize=(12, 5))

        # Display the original image
        plt.subplot(1, 3, 1)
        if original_images[i].ndim == 3:  # Check if the image is color
            plt.imshow(cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original_images[i], cmap='gray')
        plt.title("Original Image")
        plt.axis("off")



        # Display the mask
        plt.subplot(1, 3, 2)
        if masks[i] is not None:
            plt.imshow(masks[i], cmap='gray')
            plt.title("Mask")
        else:
            plt.text(0.5, 0.5, "No Mask Available", ha='center', va='center', fontsize=12)
        plt.axis("off")

        # Display the processed image
        plt.subplot(1, 3, 3)
        if processed_images[i].ndim == 3:  # Check if the image is color
            plt.imshow(cv2.cvtColor(processed_images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(processed_images[i], cmap='gray')
        plt.title("Processed Image")
        plt.axis("off")

        plt.show()

if __name__ == "__main__":
    data_directory = r'C:\Users\eyabe\Downloads\archive\COVID-19_Radiography_Dataset'
    processed_image_file = "preprocessed_images.npy"
    mask_file = "preprocessed_masks.npy"
    visualize_preprocessed(data_directory, processed_image_file, mask_file)
