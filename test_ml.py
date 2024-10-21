import pytest
# TODO: add necessary import
from fastapi.testclient import TestClient
from main import app

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    Test the root GET request, should return a welcome message and status code 200.
    """
    # Your code here
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the model inference API!"}


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    Test the POST request on the /data/ endpoint with valid data, should return status code 200.
    """
    # Your code here
    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/data/", json=data)
    assert response.status_code == 200
    assert "result" in response.json()


# TODO: implement the third test. Change the function name and input as needed
def test_three():
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
    response = client.post("/data/", json=invalid_data)
    assert response.status_code == 422

