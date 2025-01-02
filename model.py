import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense

def load_data():
    """Loads the diabetes dataset."""
    csv_path = os.path.join('resources', 'diabetes_df.csv')
    diabetes_df = pd.read_csv(csv_path)
    return diabetes_df

def preprocess_data(diabetes_df):
    """Preprocesses the data by scaling and balancing."""
    # Scale BMI
    scaler = RobustScaler()
    diabetes_df['BMI'] = scaler.fit_transform(diabetes_df[['BMI']])
    
    # Save the scaler for future use
    with open('bmi_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

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
    """Builds and compiles the Sequential model."""
    model = Sequential([
        Dense(16, input_shape=(input_shape,), activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics."""
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
    # Load the dataset
    diabetes_df = load_data()
    
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
    
    return model  # Ensure this line is present to return the trained model

def load_trained_model(model_path='diabetes_model.h5'):
    """Loads the trained model."""
    return load_model(model_path)

def preprocess_input(data):
    """Preprocesses a single input sample for prediction."""
    # Check the input shape
    if len(data) != 21:
        raise ValueError("Incorrect input shape. Expected 21 features.")
    
    # Load the BMI scaler
    with open('bmi_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale the BMI column (assuming it is the 4th feature, index 3)
    data = np.array(data, dtype=np.float32)
    data[3] = scaler.transform([[data[3]]])[0, 0]
    
    return data

def predict(data, model):
    """Makes a prediction for a single input sample."""
    # Preprocess the input data
    input_data = preprocess_input(data)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Make a prediction
    prediction = model.predict(input_data)
    return 'Diabetes' if prediction[0] >= 0.5 else 'No Diabetes'

if __name__ == "__main__":
    main()
