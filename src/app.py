import pickle
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Cargar el modelo y el escalador
model = pickle.load(open("./models/model.pkl", "rb"))
scaler = pickle.load(open("./models/scaler.pkl", "rb"))

@app.get("/")
def greet(name: str = "World"):
    return {"message": f"Hello, {name}!"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
def predict(data: list[float]):
    # Convertir los datos de entrada en un DataFrame
    X = [{
        f'X{i+1}': x 
        for i, x in enumerate(data)
    }]
    df = pd.DataFrame.from_records(X)

    # Escalar las características usando el scaler previamente guardado
    df_scaled = scaler.transform(df)

    # Realizar la predicción
    pred = model.predict(df_scaled)

    # Guardar nuevos puntos de datos para futura recopilación
    new_data = df.copy()
    new_data['Prediction'] = int(pred[0])
    new_data.to_csv('new_data_points.csv', mode='a', header=False, index=False)  # Append new data

    return {"Prediction": int(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, port=3000)
