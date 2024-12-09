import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from evaluation import plot_training_history, plot_confusion_matrix, plot_metrics_comparison

# Updated dataset paths
covid_dir = "COVID-19_Radiography_Dataset/COVID/images"
normal_dir = "COVID-19_Radiography_Dataset/Normal/images"

# Function to load images from a folder
def load_images_from_folder(folder, label, img_size=(256, 256)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        # Skip directories and hidden files
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    print(f"Loaded {len(images)} images from {folder}")
    return images, labels

def main():
    # Load and preprocess images
    print("Loading and preprocessing images...")
    covid_images, covid_labels = load_images_from_folder(covid_dir, label=1)
    normal_images, normal_labels = load_images_from_folder(normal_dir, label=0)

    # Combine and convert to numpy arrays
    images = np.array(covid_images + normal_images)
    labels = to_categorical(np.array(covid_labels + normal_labels))

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Define the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

    model = Sequential([
        Input(shape=(256, 256, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Starting model training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Predict and plot confusion matrix
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'COVID'])

if __name__ == "__main__":
    main()
