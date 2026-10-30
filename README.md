# Leukemia Detection
# Leukemia Detection Using Deep Learning

An experimental deep learning project that classifies blood-cell images using two convolutional neural network architectures: **ResNet50** and **EfficientNetV2-S**.

## Project Overview

This project includes two Jupyter notebooks for training deep learning models and a Streamlit application for running predictions and comparing model outputs.

The project explores how different CNN architectures perform on leukemia classification using blood-cell images.

## Models

- **ResNet50:** A convolutional neural network for image classification.
- **EfficientNetV2-S:** An efficient deep learning architecture for image classification.
- **Model Comparison:** The application allows users to run both models on an image and compare their predictions.

## Project Structure

- `Leukemia_ResNet50.ipynb` — Notebook for training the ResNet50 model.
- `Leukemia_EfficientNetV2.ipynb` — Notebook for training the EfficientNetV2-S model.
- `app.py` — Streamlit application for image classification and model comparison.
- `README.md` — Project documentation.

## Why Are the Trained Models Not Included?

The trained models are not uploaded to this repository because their `.keras` files are too large for convenient storage on GitHub.

Instead, the training code is provided in the `.ipynb` Jupyter notebooks. Running the notebooks through the training and model-saving steps generates the trained models and saves them in `.keras` format.

Once generated, the model files can be placed in the same directory as `app.py`, allowing the application to load and use them for predictions.

The application expects the following filenames:

- `Resnet50.keras`
- `efficientnetv2s.keras`

**Important:** Run the appropriate notebook through its model-saving step before launching the application. The saved model filenames must match the filenames expected by `app.py`.

## Dataset

The notebooks use the Leukemia Classification dataset available on Kaggle. Please review the dataset's license and terms before downloading or redistributing it.

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Jupyter Notebook

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/SaifWael9/leukemia-detection.git
cd leukemia-detection
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install tensorflow streamlit numpy pillow
```

Additional dependencies may be required depending on the notebook code.

### 3. Train the Models

Open the appropriate `.ipynb` notebook in Google Colab or Jupyter Notebook and run the required cells.

After completing the model training and saving steps, the trained models will be saved in `.keras` format.

### 4. Prepare the Model Files

Make sure the generated files are named exactly:

- `Resnet50.keras`
- `efficientnetv2s.keras`

Place them in the same directory as `app.py`.

### 5. Launch the Application

```bash
streamlit run app.py
```

The application can then load the saved models and generate predictions for uploaded images.

## Limitations and Disclaimer

This project is intended for educational and research purposes only. It has not been established as a clinically validated diagnostic system.

Predictions and confidence scores must not be used to diagnose or rule out leukemia or to make medical decisions. Clinical use would require rigorous independent validation and evaluation by qualified medical professionals.

## Future Improvements

- Add comprehensive evaluation metrics and confusion matrices.
- Improve reproducibility and dependency management.
- Add automated tests and application error handling.
- Evaluate model performance on independent datasets.
