# Next Word Prediction Using LSTM

This project focuses on developing a deep learning model to predict the next word in a given sequence of words. The model leverages a Long Short-Term Memory (LSTM) network, which is particularly effective for sequence prediction tasks. Additionally, we have built a user-friendly interface using Streamlit for real-time predictions.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [LSTM Model Architecture](#lstm-model-architecture)
5. [Deployment](#deployment)
6. [Installation and Usage](#installation-and-usage)
7. [Future Enhancements](#future-enhancements)

---

## Introduction
Predicting the next word in a sequence has applications in various domains, such as predictive text input, language modeling, and AI-driven writing assistance. By using Shakespeare's "Hamlet," this project challenges the model with complex, rich text, ensuring robust learning and generalization.

---

## Dataset
We used the text of Shakespeare's "Hamlet" as the dataset. This text offers a challenging vocabulary and structure, making it ideal for training an LSTM model on next-word prediction tasks.

---

## Project Workflow

1. **Data Collection:**
   - The dataset was sourced from a publicly available text version of Shakespeare’s "Hamlet."

2. **Data Preprocessing:**
   - Tokenized the text to convert it into individual words.
   - Created sequences of words with a fixed window size.
   - Padded sequences to ensure uniform lengths.
   - Split the dataset into training and testing sets.

3. **Model Building:**
   - Constructed an LSTM model with:
     - An embedding layer for word representation.
     - Two LSTM layers for sequence learning.
     - A dense output layer with a softmax activation function for next-word prediction.

4. **Model Training:**
   - Utilized early stopping to monitor validation loss and prevent overfitting.

5. **Deployment:**
   - Developed a Streamlit web application for real-time predictions.

---

## LSTM Model Architecture

Below is a flowchart depicting the LSTM model architecture:

```plaintext
Input Sequences (Tokenized & Padded) --> Embedding Layer --> First LSTM Layer --> Second LSTM Layer --> Dense Layer with Softmax Activation --> Predicted Word
```

### Flowchart Representation:
![LSTM Architecture]([LSTM_Flowchart.png](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwuRVVFzUh7lBaeY5Ws15C9pp5fKAYL_KkBg&s))

---

## Deployment
The model is deployed using a Streamlit application. Users can input a sequence of words, and the app predicts the next word in real-time. The interface is designed to be intuitive and responsive, ensuring a seamless user experience.

---

## Installation and Usage

### Prerequisites
- Python 3.7+
- Required libraries: TensorFlow, Keras, NumPy, Pandas, Streamlit, NLTK

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/next-word-prediction-lstm.git
   cd next-word-prediction-lstm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the app in your browser at `http://localhost:8501`.

---

## Future Enhancements
- Expand the dataset to include multiple literary works.
- Introduce beam search for more accurate predictions.
- Implement multilingual support for non-English texts.
- Improve the UI/UX of the Streamlit application.

---

## Acknowledgements
- Shakespeare's "Hamlet" text from [Project Gutenberg](https://www.gutenberg.org/).
- TensorFlow/Keras libraries for model implementation.
- Streamlit for creating an interactive web application.

---

Happy Coding! 🎉

