Human Voice Classification and Clustering

🎤 Human Voice Classification and Clustering is a Machine Learning project that focuses on analyzing audio signals to classify human voices (e.g., male/female or speaker identification) and group similar voices using clustering techniques.

🚀 Project Overview

Objective: Build an end-to-end system that extracts features from voice samples, classifies them using supervised ML models, and groups voices using unsupervised clustering.

Key Techniques:

Feature Extraction: MFCCs (Mel-frequency cepstral coefficients), Chroma, Spectral features

Classification Models: Random Forest, SVM, Neural Networks 

Clustering Models:K-Means, DBSCAN

Applications:

Gender classification

Speaker verification

Voice-based biometric authentication

Audio pattern analysis

🛠️ Tech Stack

Language: Python

Libraries:

librosa (audio processing)

scikit-learn (ML models, clustering)

pandas, numpy (data handling)

Project Structure
Human-Voice-Classification-and-Clustering/
│── data/               # Dataset (audio files, features)
│── data_preprocessing/          # processing data
│── eda_analysis/                # Python scripts for visualizing
│── model/             # preparing model
│── model1/            # save model
|__ rf_gender_pipeline.pkl #Saved model
│── app/                # (Optional) Streamlit app for demo
│── requirements.txt    # Dependencies
│── README.md           

<img width="980" height="837" alt="image" src="https://github.com/user-attachments/assets/00a4205e-5fe5-49a8-9409-15732a92210e" />
<img width="997" height="572" alt="image" src="https://github.com/user-attachments/assets/267547c9-e528-457f-bfbf-2ab872337ef1" />


matplotlib, seaborn (visualization)

Tools: Jupyter Notebook / VS Code, GitHub
