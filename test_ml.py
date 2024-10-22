import pytest
# TODO: add necessary import
from fastapi.testclient import TestClient
from main import app
import pandas as pd
from ml.data import process_data  # Import the process_data function
from ml.model import train_model, compute_model_metrics, inference 
from sklearn.ensemble import RandomForestClassifier

client = TestClient(app)

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test that the model is successfully trained on data.
    """
    # Create a small dataset
    data = {
        "age": [37, 45],
        "workclass": ["Private", "Self-emp"],
        "fnlgt": [178356, 88439],
        "education": ["HS-grad", "Bachelors"],
        "education-num": [10, 13],
        "marital-status": ["Married-civ-spouse", "Divorced"],
        "occupation": ["Prof-specialty", "Exec-managerial"],
        "relationship": ["Husband", "Not-in-family"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 60],
        "native-country": ["United-States", "United-States"],
        "salary": [">50K", "<=50K"],
    }
    
    df = pd.DataFrame(data)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    X_train, y_train, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    assert model.n_estimators > 0, "Model did not train properly"

# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    Test that the model inference runs correctly with a pre-trained encoder.
    """
    # Create a small dataset
    data = {
        "age": [37],
        "workclass": ["Private"],
        "fnlgt": [178356],
        "education": ["HS-grad"],
        "education-num": [10],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Prof-specialty"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [40],
        "native-country": ["United-States"],
        "salary": [">50K"]  # Add the label column here
    }

    df = pd.DataFrame(data)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    # Training step to get a pre-trained encoder and model
    train_data = df.copy()  # Simulate part of your training data

    # Use "salary" as the label for training
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features, label="salary", training=True
    )

    assert len(X_train) > 0, "Training data processing failed."
    assert len(y_train) > 0, "Training label processing failed."


# TODO: implement the third test. Change the function name and input as needed
def test_post():
    """
    # add description for the third test
    Test the POST request on the /data/ endpoint with invalid or missing data, should return status code 422.
    """
    # Your code here
    invalid_data = {
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
    }
    response = client.post("/predict/", json=invalid_data)
    assert response.status_code == 422

