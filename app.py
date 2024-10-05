import pickle
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd

app = FastAPI()

model = pickle.load(open("./models/model.pkl", "rb"))

@app.get("/")
def greet(name: str = "World"):
    return {"message": f"Hello, {name}!"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
def predict(data:list[float]):
    X = [{
        f'X{i+1}': x 
        for i, x in enumerate(data)
    }]
    df = pd.DataFrame.from_records(X)
    pred = model.predict(df)

    # Save new data points for future collection
    new_data = df.copy()
    new_data['Prediction'] = int(pred[0])
    new_data.to_csv('new_data_points.csv', mode='a', header=False, index=False)  # Append new data

    return {"Prediction":int(pred[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=3000)