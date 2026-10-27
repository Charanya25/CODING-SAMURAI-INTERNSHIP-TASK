# Facial Expression Recognition

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify facial expressions from images.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Streamlit
- Pillow
- Matplotlib
- Scikit-learn

## Expression Categories
The model predicts seven expression categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Performance
- Test Accuracy: 54.19%
- Test Loss: 1.2002

These are the results obtained during testing. Predictions may be incorrect.

## Project Files
- `app.py` — Streamlit application
- `facial_expression_model.keras` — Trained model
- `facial_expression_recognition.ipynb` — Training notebook
- `requirements.txt` — Required Python packages

## How to Run
1. Install Python.
2. Open Command Prompt in the project folder.
3. Run `python -m pip install -r requirements.txt`.
4. Run `python -m streamlit run app.py`.
5. Open the local URL displayed in the terminal.

## Disclaimer
This is an educational project. The predicted expression does not establish a person's actual emotional state.
