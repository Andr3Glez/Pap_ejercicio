import requests

data = list(range(23))

response = requests.post("http://localhost:3000/predict", json=data)

prediction = response.json()["Prediction"]

print(prediction)