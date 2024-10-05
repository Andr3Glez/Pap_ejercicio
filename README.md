# PAP Actividad 4: MLOps 

_Description:_
Considering the main points of MLOps (Training, Deploying, Monitoring), create the following:

- A GitHub repository with a README file and python's .gitignore template.
- A training script that saves the required artifacts to use the model.
- An API with a health check endpoint and a `/predict` route that takes in the independent variables and returns the prediction, saving the new data points in a separate dataset as- part of our data collection step.
- Apply Chi-squared and Kolmogorov-Smirnoff tests in order to determine if the dataset distributions are significantly different.
- Just throw a print statement when the hypothesis tests fail.
- Create a python script that sends the requests to your app.
- Using the `credit_pred.csv` dataset, create a `Y` column using your model's predictions.
- Upload the GitHub repository link including these scripts and the `credit_pred.csv` dataset with your predictions.


## Comenzando 🚀


### Pre-requisitos 📋

_Para instalar librerias_

```
pip install -r requirements.txt
```


### Instalación 🔧

_Clona el repo_
```
git clone https://github.com/Andr3Glez/Pap_ejercicio.git
```
_instalar librerias_
```
pip install -r requirements.txt
```
_Ejecutar el train.csv_
```
python src/train.py
```
_Ejecutar el app.csv_
```
python src/app.py
```

### Uso de API 📖

La aplicación tiene las siguientes rutas disponibles:

_1. Saludo_

**Ruta**: `/`

- **Método**: GET
- **Descripción**: Devuelve un saludo.
- **Ejemplo de uso**:

```bash
   curl "http://localhost:3000/?name=TuNombre"
```
_2. Salud_

**Ruta**: `/health`

- **Método**: GET
- **Descripción**: Verifica el estado de la api.
- **Ejemplo de uso**:

```bash
   curl "http://localhost:3000/health"
```
- **Respuesta**:
```json
   {
    "status": "OK"
    }
```
_3. Predicción_

**Ruta**: `/predict`

- **Método**: GET
- **Descripción**: Prediccion del modelo con datos de entrada.

- **Respuesta**:
```json
   {
     "Prediction": 1
    }
```

## Autores ✒️

* **Sergio Dueñas** - [SergioDuenass](https://github.com/SergioDuenass)
* **Andre González** -  [Andr3Glez](https://github.com/Andr3Glez)



---