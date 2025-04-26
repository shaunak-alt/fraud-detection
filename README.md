Fraud Detection System Using Genetic Algorithms and Machine Learning

Project Overview
This project implements a Fraud Detection System using Genetic Algorithms for feature selection and a Random Forest Classifier for fraud classification. The objective is to build an efficient, accurate, and interpretable fraud detection model that reduces feature redundancy and improves predictive performance.

Features
1. Data Preprocessing: Handles missing values, normalizes features, and applies log transformations to reduce skewness.
2. Feature Selection: Uses Genetic Algorithm (GA) to optimize and select the most important features.
3. Model Training: Trains a Random Forest Classifier on the selected features.
4. Evaluation: Provides metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
5. Fraud Prediction: Includes functionality to predict fraud likelihood for new transactions.

Technologies Used
1. Python
2. Pandas
3. NumPy
4. scikit-learn
5. DEAP (Distributed Evolutionary Algorithms in Python)

Project Structure
├── fraud.csv                  # Dataset (not included in repo - upload manually)
├── genetic_fraud_detection.py  # Main code file
├── README.md                   # Project documentation
