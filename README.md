# Cultural Algorithms & ML for Image Processing

## 📜 Overview

This project implements **Cultural Algorithms** combined with **Machine Learning** techniques to solve image processing tasks, specifically focusing on segmentation and optimization. The project leverages evolutionary computation to iteratively improve results, simulating the evolution of a population guided by belief spaces.

## 🚀 Features

- **Cultural Algorithm Implementation:** Optimizes solutions based on evolutionary principles.
- **Image Preprocessing:** Efficient preprocessing of images for segmentation tasks.
- **Visualization Tools:** Scripts to visualize preprocessed data and segmentation results.
- **Modular Codebase:** Clean, well-structured Python scripts for easy maintainability.
- **Fitness Evaluation:** Custom fitness functions to assess the quality of solutions.

## 🛠️ Technologies Used

- **Python 3.8+**
- **NumPy** for numerical operations
- **Matplotlib** for visualization
- **Jupyter Notebook** for interactive experimentation

## 📁 Project Structure

```
Cultural_Algorithms_Image_Processing/
│
├── .ipynb_checkpoints/           # Jupyter notebook checkpoints
├── __pycache__/                  # Python cache files
│
├── Untitled.ipynb                # Jupyter notebook for interactive analysis
├── Visualize_preprocessed.py     # Script to visualize preprocessed images
├── belief_space.py               # Implements belief space logic for the cultural algorithm
├── evaluation.py                 # Fitness evaluation functions
├── evolution.py                  # Core evolutionary algorithm logic
├── fitness.py                    # Defines the fitness function
├── main.py                       # Entry point for executing the project
├── population.py                 # Handles population initialization and updates
├── preprocess_and_save.py        # Script to preprocess and save image data
│
├── preprocessed_images.npy       # Saved preprocessed image data
└── preprocessed_masks.npy        # Saved preprocessed mask data
```

## ⚙️ Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/username/cultural-algorithms-ml.git
   cd cultural-algorithms-ml
   ```

2. **Set Up the Environment:**

   Ensure you have Python 3.8+ and install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Jupyter Notebook (Optional):**

   ```bash
   pip install jupyter
   ```

## 📝 How to Use

### 1. **Preprocess the Images**

Run the preprocessing script to prepare image data:

```bash
python preprocess_and_save.py
```

### 2. **Run the Cultural Algorithm**

Execute the main script to perform the evolutionary optimization:

```bash
python main.py
```

### 3. **Visualize the Results**

Generate visualizations of the preprocessed images and results:

```bash
python Visualize_preprocessed.py

```
![R3](https://github.com/user-attachments/assets/fe14d661-d392-4d0b-9ccd-99650b5672e0)


### 4. **Interactive Analysis**

Use the `Untitled.ipynb` notebook for interactive experiments:

```bash
jupyter notebook Untitled.ipynb
```

## 🖼️ Sample Results
![A6](https://github.com/user-attachments/assets/b32a1984-7f43-4c67-a9a4-8a2c5bbf8284)
![A5](https://github.com/user-attachments/assets/612a532b-8297-4c07-89e4-9204a919f804)
![A4](https://github.com/user-attachments/assets/12401f74-16c8-4a50-9479-824170ae6b56)
![A3](https://github.com/user-attachments/assets/463cfc86-ffbd-468c-8e0b-0038456473fd)
![A2](https://github.com/user-attachments/assets/d177979d-c0bf-4975-8dab-475a0b47cb39)
![A1](https://github.com/user-attachments/assets/8ad3f502-fa6a-4b6c-8f84-2f4a79ab4b35)


![R1](https://github.com/user-attachments/assets/272a5109-c840-4f93-be6c-95d0b65353b1)
![R2](https://github.com/user-attachments/assets/4bbf6ef0-f48f-4f59-b4d2-7ddc29bafbd6)
![4b6db7dc-58ed-4c7b-8a44-5c1690a26b3b](https://github.com/user-attachments/assets/e5324a1c-1a7a-4e59-82d4-54f64f8fd3b4)



![Capture d’écran 2024-12-09 175016](https://github.com/user-attachments/assets/00343faa-b508-4b0c-ac81-131428e3844a)

---
![comp](https://github.com/user-attachments/assets/34f0acac-2ebf-41ae-8666-2dea33b9815a)
![Capture d’écran 2024-12-05 192036](https://github.com/user-attachments/assets/2cb22268-854c-4b7c-a459-31050c2fe93d)


## 📖 How the Cultural Algorithm Works

1. **Belief Space**: Represents shared knowledge guiding the population's evolution.
2. **Population Space**: Individuals evolve based on both their personal experience and the belief space.
3. **Evaluation**: A fitness function evaluates the performance of each individual.
4. **Evolution**: Individuals are selected, mutated, and evolved over generations.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Project**.
2. **Create a New Branch**: `git checkout -b feature-branch`
3. **Make Changes and Commit**: `git commit -m "Add new feature"`
4. **Push to the Branch**: `git push origin feature-branch`
5. **Submit a Pull Request**.

---


- The open-source community for their resources and inspiration.

---

### 🌟 Happy Coding!

