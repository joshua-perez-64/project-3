import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

def load_data():
    # Load data
    csv_path = os.path.join('resources', 'diabetes_data.csv')
    diabetes_df = pd.read_csv(csv_path)
    return diabetes_df

def preprocess_data(diabetes_df):
    # Scale BMI
    scaler = RobustScaler()
    diabetes_df['BMI'] = scaler.fit_transform(diabetes_df[['BMI']])

    # Split the data into features (X) and target (y)
    X = diabetes_df.drop(columns='Diabetes_binary')
    y = diabetes_df['Diabetes_binary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Undersample to create a balanced dataset
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    return X_resampled, y_resampled, X_test, y_test

def build_model(input_shape):
    # Build a Sequential model
    model = Sequential()
    model.add(Dense(16, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    testing_predictions = (model.predict(X_test) > 0.5).astype("int32")
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, testing_predictions)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, testing_predictions, labels=[1, 0])
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    # Classification report
    class_report = classification_report(y_test, testing_predictions, labels=[1, 0])
    print(f'Classification Report:\n{class_report}')
    
    return test_accuracy

def main():
    # Load your dataset (replace 'your_data.csv' with your actual data file)
    diabetes_df = pd.read_csv('resources\diabetes_df.csv')
    
    # Preprocess the data
    X_resampled, y_resampled, X_test, y_test = preprocess_data(diabetes_df)
    
    # Build the model
    model = build_model(X_resampled.shape[1])
    
    # Train the model
    model.fit(X_resampled, y_resampled, validation_data=(X_test, y_test), epochs=30, batch_size=32)
    
    # Save the model
    model.save('diabetes_model.h5')
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

def load_trained_model(model_path='diabetes_model.h5'):
    return load_model(model_path)

# Preproccess input
def preprocess_input(data):
    diabetes_df = pd.read_csv('resources\diabetes_df.csv')
    # Scale BMI
    scaler = RobustScaler()
    diabetes_df['BMI'] = scaler.fit_transform(diabetes_df[['BMI']])
    return data

def predict(data, model):
    # Ensure data is in numpy array format
    input_data = np.array([data])

    # Predict using model
    prediction = model.predict(input_data)

    # Return the predictions
    return 'Diabetes' if prediction[0] >= 0.5 else 'No Diabetes'

if __name__ == "__main__":
    main()
