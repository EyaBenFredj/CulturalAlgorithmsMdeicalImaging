import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.src.models.sequential import Sequential
from keras.src.utils.numerical_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation import plot_training_history, plot_confusion_matrix


def plot_metrics_comparison(baseline_metrics, ca_metrics, metric_names):
    import matplotlib.pyplot as plt
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, baseline_metrics, width, label='Baseline Model')
    plt.bar(x + width/2, ca_metrics, width, label='Cultural Algorithm Model')

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Baseline and CA-Based Model Metrics')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.show()


def load_data(file_name, use_memory_map=False, subset_size=None):
    if use_memory_map:
        data = np.load(file_name, mmap_mode='r')
    else:
        data = np.load(file_name, allow_pickle=True)

    if subset_size is not None:
        data = data[:subset_size]

    return data


def main():
    # Load and preprocess data
    if os.path.exists("preprocessed_images.npy") and os.path.exists("preprocessed_masks.npy"):
        print("Loading preprocessed data from disk...")
        start_time = time.time()
        preprocessed_images = load_data("preprocessed_images.npy", use_memory_map=True, subset_size=1000)
        preprocessed_masks = load_data("preprocessed_masks.npy", use_memory_map=True, subset_size=1000)
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    else:
        print("No preprocessed data found. Please run the preprocessing script (preprocess_and_save.py) first.")
        return

    preprocessed_images = np.array(preprocessed_images, dtype=np.float32) / 255.0
    labels = np.random.randint(0, 2, size=(len(preprocessed_images),))  # Example binary classification
    labels = to_categorical(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(preprocessed_images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Define data augmentation
    data_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    data_augmentation.fit(X_train)

    # Define the model
    model = Sequential([
        Input(shape=(preprocessed_images.shape[1], preprocessed_images.shape[2], 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(labels.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Train the model with data augmentation and callbacks
    print("Starting model training...")
    history = model.fit(data_augmentation.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_val, y_val), epochs=50,
                        callbacks=[early_stopping, reduce_lr, checkpoint])

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Predict and plot confusion matrix
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    plot_confusion_matrix(y_true, y_pred, labels=['Class 0', 'Class 1'])

    # Example metric comparison plot
    baseline_metrics = [0.85, 0.80, 0.82, 0.81]
    ca_metrics = [0.88, 0.83, 0.85, 0.84]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    plot_metrics_comparison(baseline_metrics, ca_metrics, metric_names)


if __name__ == "__main__":
    main()
