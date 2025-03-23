# Pneumonia-Disease-Detection

Pneumonia-Disease-Detection is a deep learning-based application that automatically classifies chest X-ray images into four categories: COVID-19, Normal, Pneumonia-Bacterial, and Pneumonia-Viral. This tool aims to assist healthcare professionals in diagnosing pneumonia types quickly and accurately.

## Project Overview

This project consists of two main components:
1. **Jupyter Notebook** - Contains the model development, training and evaluation
2. **Streamlit App** - Provides the user interface for pneumonia classification

The trained model (best_model.h5) connects these components, being developed in the notebook and deployed in the Streamlit app.

## Features

- **Automated X-ray Classification**: Uses a custom CNN model to classify chest X-ray images
- **User Authentication**: Secure login system with different access levels for doctors and patients
- **Patient Dashboard**: Allows patients to upload images and track diagnosis history
- **Doctor Interface**: Enables doctors to review predictions, add notes, and prescribe treatments
- **Diagnosis Tracking**: Monitors the status of reports from submission to diagnosis
- **Modern UI/UX**: Responsive design with intuitive navigation and visual feedback

## Screenshots

### Application Interface

![Screenshot 2025-03-22 222342](https://github.com/user-attachments/assets/7c1cf0f0-50ba-47ff-b409-a715a9ac1110)

### Patient Dashboard

![Screenshot 2025-03-23 004602](https://github.com/user-attachments/assets/59d04c6f-919f-4360-8131-48f233011517)             ![Screenshot 2025-03-23 004917](https://github.com/user-attachments/assets/80f683d6-015e-4d20-8f13-1dabbc1cdd32)

### Diagnosis Tracking

![Screenshot 2025-03-23 010736](https://github.com/user-attachments/assets/32237ad1-2af5-434c-9a00-0fb9a3cfa603)

### Doctor Dashboard

![Screenshot 2025-03-23 005037](https://github.com/user-attachments/assets/e5efb137-c4ce-4fb4-89a1-d8a0edcf7776)

## Performance Metrics

Our model achieves the following performance on the test dataset:
![Screenshot 2025-03-22 221727](https://github.com/user-attachments/assets/8e926c3d-9690-41b4-b1a9-a87a7990a879)

## Model Architecture

The classification model is a custom Convolutional Neural Network (CNN) with:

- Three convolutional blocks with increasing filter sizes (32, 64, 128)
- Batch normalization after each convolutional layer
- Dropout layers to prevent overfitting
- Dense layers for final classification
- Training optimization with early stopping and learning rate reduction

## Dataset

The model is trained on a custom dataset named "Processed Data set" which is organized as follows:

- **Dataset Structure**:
  - `test/` - Test images for model evaluation
  - `train/` - Training images for model learning
  - `val/` - Validation images for hyperparameter tuning

- **Classes**:
  - COVID-19
  - Normal
  - Pneumonia-Bacterial
  - Pneumonia-Viral

Each folder (test, train, val) contains chest X-ray images separated into these four diagnostic categories. The images are used to train the model to distinguish between normal lungs, COVID-19 cases, and different types of pneumonia.

After downloading the repository, the dataset should be placed in the project root directory for the model training notebook to access it correctly.

## Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Web Application**: Streamlit
- **Database**: SQLite
- **Development Environment**: Jupyter Notebook
- **Visualization**: Matplotlib, Seaborn
- **Data Analysis**: Pandas, NumPy, Scikit-learn

## Getting Started

### Prerequisites

```
tensorflow>=2.5.0
keras>=2.5.0
streamlit>=1.0.0
jupyter>=1.0.0
pandas>=1.3.0
numpy>=1.19.5
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.1
```

### Installation

1. Clone the repository and download the dataset:
   ```
   git clone https://github.com/dishajain123/Pneumonia-Disease-Detection.git
   cd Pneumonia-Disease-Detection
   ```

   Download the "Processed Data set" and place it in the project root directory.
   The dataset should contain three folders: `test`, `train`, and `val`, each with four subfolders for the different classes.
   

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install tensorflow keras streamlit jupyter pandas numpy scikit-learn matplotlib seaborn
   ```

4. Open the Jupyter notebook to view the model development process:
   ```
   jupyter notebook
   ```
   Then open the `PDD.ipynb` file

5. Launch the Streamlit application:
   ```
   streamlit run app.py
   ```

Note: The model file `best_model.h5` should be in the same directory as `app.py` for the application to work properly.

## Usage

1. **Login/Register**: Access the system as a doctor or patient
2. **Upload Image**: Submit a chest X-ray for analysis
3. **View Prediction**: See the model's classification and confidence score
4. **Doctor Review**: Doctors can confirm or adjust the diagnosis and add notes
5. **Track Status**: Patients can monitor the progress of their diagnosis

## Project Files

- **PDD.ipynb**: Jupyter notebook containing the model development code
- **app.py**: Streamlit application for the user interface
- **best_model.h5**: Trained CNN model for pneumonia classification
- **class_labels.json**: Class labels used for prediction
- **pneumonia_app.db**: SQLite database for storing patient records and diagnoses

## Model Training

The model was trained on a dataset of chest X-ray images divided into four categories. Training involved:

- Data augmentation to increase training sample diversity
- Class weighting to handle imbalanced data
- Validation-based early stopping to prevent overfitting
- Learning rate reduction to optimize convergence

The model is trained with a batch size of 16 for up to 25 epochs, utilizing grayscale images of size 150x150 pixels.

## Future Improvements

- Integration with hospital management systems
- Mobile application for remote diagnosis
- Additional classification categories for other lung conditions
- Enhanced explainability features to highlight regions of interest in X-rays
- Expanded database functionality for long-term patient tracking

## Contributors

- [Disha Jain](https://github.com/dishajain123)

## Acknowledgments

- Special thanks to the open-source community for providing tools and libraries
- Acknowledgment to medical institutions for guidance on pneumonia classification
