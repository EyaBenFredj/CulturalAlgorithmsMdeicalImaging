import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from keras.src.models.sequential import Sequential
from keras.src.utils.numerical_utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


# Function to load images and labels
def load_images_and_labels(dataset_path, subset_size=500):
    images = []
    labels = []

    # Define class labels based on folder names
    class_labels = {"COVID": 0, "Lung_Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}

    for class_name, class_index in class_labels.items():
        class_folder = os.path.join(dataset_path, class_name)
        image_folder = os.path.join(class_folder, "images")

        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for i, img_file in enumerate(image_files):
            if subset_size and i >= subset_size:
                break

            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                images.append(img)
                labels.append(class_index)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = to_categorical(labels, num_classes=4)

    return np.expand_dims(images, -1), labels

# Function to build the CNN model
def build_model(input_shape, num_classes, learning_rate=0.001, num_filters=32, dropout_rate=0.3):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(num_filters, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to display predictions
def display_predictions(images, y_true, y_pred, class_names, num_samples=10):
    plt.figure(figsize=(20, 10))

    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        actual_label = class_names[y_true[i]]
        predicted_label = class_names[y_pred[i]]
        color = 'green' if actual_label == predicted_label else 'red'
        plt.title(f"Actual: {actual_label}\nPred: {predicted_label}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

# Function to compare metrics
def plot_metrics_comparison(baseline_metrics, ca_metrics, metric_names):
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_metrics, width, label='Baseline')
    plt.bar(x + width/2, ca_metrics, width, label='Cultural Algorithm')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Metrics Comparison')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.show()

# Cultural Algorithm for Hyperparameter Optimization
def cultural_algorithm_optimization():
    import random

    population = [
        {'learning_rate': random.uniform(0.0001, 0.01), 'num_filters': random.choice([16, 32, 64]), 'dropout_rate': random.choice([0.2, 0.3, 0.5])}
        for _ in range(10)
    ]

    best_params = None
    best_accuracy = 0

    for i, params in enumerate(population):
        print(f"Evaluating candidate {i+1}: {params}")
        model = build_model(images.shape[1:], labels.shape[1], **params)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Best Hyperparameters Found: {best_params}, Best Accuracy: {best_accuracy:.4f}")
    return best_params

# Main function
def main():
    dataset_path = r"C:\Users\eyabe\PycharmProjects\CulturalAlgorithmsMdeicalImaging\COVID-19_Radiography_Dataset"

    global images, labels, X_train, X_val, X_test, y_train, y_val, y_test
    images, labels = load_images_and_labels(dataset_path, subset_size=500)

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Train the Baseline Model with fewer epochs
    print("Training Baseline Model...")
    baseline_model = build_model(images.shape[1:], labels.shape[1])
    history_baseline = baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=1)])

    y_pred_baseline = baseline_model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Evaluate the Baseline Model
    accuracy_baseline = accuracy_score(y_true, y_pred_baseline)
    precision_baseline = precision_score(y_true, y_pred_baseline, average='weighted', zero_division=1)
    recall_baseline = recall_score(y_true, y_pred_baseline, average='weighted')
    f1_baseline = f1_score(y_true, y_pred_baseline, average='weighted')

    # Simulate Cultural Algorithm Optimization
    print("\nOptimizing Hyperparameters with Cultural Algorithm...")
    best_params = cultural_algorithm_optimization()
    ca_model = build_model(images.shape[1:], labels.shape[1], **best_params)
    history_ca = ca_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
    y_pred_ca = ca_model.predict(X_test).argmax(axis=1)

    # Metrics Comparison
    accuracy_ca = accuracy_score(y_true, y_pred_ca)
    precision_ca = precision_score(y_true, y_pred_ca, average='weighted', zero_division=1)
    recall_ca = recall_score(y_true, y_pred_ca, average='weighted')
    f1_ca = f1_score(y_true, y_pred_ca, average='weighted')

    # Plot Confusion Matrix for Baseline Model
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    plot_confusion_matrix(y_true, y_pred_baseline, labels=class_names)

    # Plot Training History for Baseline Model
    plot_training_history(history_baseline)

    # Plot Metrics Comparison
    baseline_metrics = [accuracy_baseline, precision_baseline, recall_baseline, f1_baseline]
    ca_metrics = [accuracy_ca, precision_ca, recall_ca, f1_ca]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    plot_metrics_comparison(baseline_metrics, ca_metrics, metric_names)

    # Display Predictions
    display_predictions(X_test, y_true, y_pred_baseline, class_names, num_samples=10)

if __name__ == "__main__":
    main()
