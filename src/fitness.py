from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import cv2

def calculate_fitness(individual, dataset, labels, model):
    """
    Calculate the fitness of an individual based on its performance on a dataset.

    Parameters:
    - individual: The individual being evaluated (contains features, preprocessing methods, etc.).
    - dataset: The dataset (images) to be evaluated on.
    - labels: Corresponding labels for the dataset.
    - model: A machine learning model to use for evaluation (e.g., a simple classifier).

    Returns:
    - fitness_score: The calculated fitness score (composite of accuracy, precision, recall, and F1).
    """
    # Preprocess dataset based on individual's preprocessing choices
    processed_dataset = []
    for image in dataset:
        # Apply the individual's preprocessing steps
        for method in individual.preprocessing:
            if method == "histogram_equalization":
                if len(image.shape) == 3:  # Convert to grayscale if image has 3 channels (e.g., RGB)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.equalizeHist(image)
            elif method == "gaussian_filter":
                image = cv2.GaussianBlur(image, (5, 5), 0)
        processed_dataset.append(image)

    processed_dataset = np.array(processed_dataset)

    # Optional: Normalize the images (scaling pixel values to [0, 1])
    processed_dataset = processed_dataset / 255.0

    # Flatten the images if required by the model (optional, depends on model)
    processed_dataset = processed_dataset.reshape(processed_dataset.shape[0], -1)

    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(processed_dataset, labels, test_size=0.2, random_state=42)

    # Train the model with the individual's features
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate fitness using a combination of metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Composite fitness score (can be weighted as needed)
    fitness_score = (accuracy + precision + recall + f1) / 4

    return fitness_score
