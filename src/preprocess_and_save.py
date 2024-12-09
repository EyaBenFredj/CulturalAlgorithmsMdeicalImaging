import os
import cv2


def load_images_and_masks(data_dir):
    images = []
    masks = []
    labels = []

    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        images_path = os.path.join(label_path, 'images')
        masks_path = os.path.join(label_path, 'masks')

        if not os.path.exists(images_path):
            continue

        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for image_name in image_files:
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            mask_path = os.path.join(masks_path, image_name)
            mask = None
            if os.path.exists(mask_path) and mask_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                mask = cv2.imread(mask_path, 0)

            images.append(image)
            masks.append(mask)
            labels.append(label_folder)

    return images, masks, labels


def preprocess_data(image, mask=None, method="none"):
    processed_image = image

    if method == "histogram_equalization":
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        processed_image = cv2.equalizeHist(processed_image)
        print("Applied histogram equalization")

    elif method == "gaussian_filter":
        # Apply a Gaussian filter (blurring)
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)
        print("Applied Gaussian filter")

    elif method == "normalize":
        # Normalize the image to range [0, 1]
        processed_image = image / 255.0
        print("Applied normalization")

    if mask is not None:
        return processed_image, mask
    else:
        return processed_image


if __name__ == "__main__":
    data_directory = r'C:\Users\eyabe\Downloads\archive\COVID-19_Radiography_Dataset'
