# @Author: Humberto Alejandro Ortega Alcocer
# @Date: 2024-Abril-8
# @Description: API para predecir el precio de una propiedad utilizando tres modelos de aprendizaje automático: Random Forest, SVM y Redes Neuronales.
# @Dependencies: joblib, tensorflow, pydantic, pandas, matplotlib, tempfile, fastapi
# @Usage: uvicorn main:app --reload
# @URL: http://localhost:8000
# @Docs: http://localhost:8000/openapi
# @Redoc: http://localhost:8000/redoc
# @License: MIT
from fastapi.middleware.cors import CORSMiddleware # Para configurar CORS
import datetime # Para obtener la fecha y hora actual
from joblib import load # Para cargar los modelos de aprendizaje automático
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error # Para calcular el error absoluto medio
from pydantic import BaseModel, Field # Para definir la clase de la propiedad y sus campos
import pandas as pd # Para trabajar con los datos de la propiedad
from fastapi import FastAPI # Para crear la aplicación de FastAPI
from fastapi.responses import FileResponse # Para servir archivos estáticos
import matplotlib # Para configurar el renderizador
import matplotlib.pyplot as plt # Para generar gráficas
import logging # Para mostrar mensajes de depuración
from constants import *

# Mensaje de depuración
logging.basicConfig(level=logging.INFO)
logging.info("Cargando modelos de aprendizaje automático...")

# Elegimos usar el renderizador "Agg" que no requiere usar el entorno gráfico
# de nuestro Sistema Operativo.
matplotlib.use('Agg')

# Carga de modelos
rf_model = load(ARCHIVO_MODELO_RF)
svm_model = load(ARCHIVO_MODELO_SVM)
nn_model = load(ARCHIVO_MODELO_RN)
nn_model_history = load(ARCHIVO_MODELO_RN_HISTORIA)
preprocessor = load(ARCHIVO_PREPROCESADOR)

logging.info("Modelos de aprendizaje automático cargados con éxito")
logging.info("Cargando datos de entrenamiento y prueba...")

# Carga de datos de entrenamiento y prueba
X_train = pd.read_csv(ARCHIVO_X_TRAIN)
X_test = pd.read_csv(ARCHIVO_X_TEST)
y_train = pd.read_csv(ARCHIVO_Y_TRAIN)
y_test = pd.read_csv(ARCHIVO_Y_TEST)

# Carga del dataset original
df = pd.read_csv(ARCHIVO_DATASET)

logging.info("Datos de entrenamiento y prueba cargados con éxito")
logging.info("Evaluando los modelos de aprendizaje automático...")

# Predecir con cada modelo
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_test)
nn_preds = nn_model.predict(preprocessor.transform(X_test)).flatten()

# Calcular el error cuadrático medio de cada modelo
rf_mse = mean_squared_error(y_test, rf_preds)
svm_mse = mean_squared_error(y_test, svm_preds)
nn_mse = mean_squared_error(y_test, nn_preds)

# Calcular el RMSE (raíz del error cuadrático medio) de cada modelo
rf_rmse = root_mean_squared_error(y_test, rf_preds)
svm_rmse = root_mean_squared_error(y_test, svm_preds)
nn_rmse = root_mean_squared_error(y_test, nn_preds)

# Calcular los intervalos de confianza de cada modelo
rf_ci = (rf_preds - y_test.squeeze()).mean() - 1.96 * (rf_preds - y_test.squeeze()).std(), (rf_preds - y_test.squeeze()).mean() + 1.96 * (rf_preds - y_test.squeeze()).std()
svm_ci = (svm_preds - y_test.squeeze()).mean() - 1.96 * (svm_preds - y_test.squeeze()).std(), (svm_preds - y_test.squeeze()).mean() + 1.96 * (svm_preds - y_test.squeeze()).std()
nn_ci = (nn_preds - y_test.squeeze()).mean() - 1.96 * (nn_preds - y_test.squeeze()).std(), (nn_preds - y_test.squeeze()).mean() + 1.96 * (nn_preds - y_test.squeeze()).std()

# Calculamos el error absoluto medio de cada modelo
rf_mae = mean_absolute_error(y_test, rf_preds)
svm_mae = mean_absolute_error(y_test, svm_preds)
nn_mae = mean_absolute_error(y_test, nn_preds)

# Calculamos el coeficiente de determinación de cada modelo
rf_r2 = rf_model.score(X_test, y_test)
svm_r2 = svm_model.score(X_test, y_test)
nn_r2 = nn_model_history.history['r2_score'][-1]

logging.info("Modelos de aprendizaje automático evaluados con éxito")
logging.info("Generando gráficas de predicciones vs valores reales...")

# Generación de gráficas
models = {'Random Forest': rf_preds, 'SVM': svm_preds, 'Neural Network': nn_preds}
axs = plt.subplots(1, 3, figsize=(15, 5))[1]

# Graficar las predicciones vs valores reales
for ax, (model_name, preds) in zip(axs, models.items()):
    ax.scatter(y_test, preds, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Predicciones')
    ax.set_title(model_name)

# Ajustar el espacio entre las gráficas
plt.tight_layout()

# Servimos la gráfica sin tocar el disco
plt.savefig('plots.png')

# Cerramos la gráfica
plt.close('all')

logging.info("Gráficas generadas con éxito")

# Inicialización de la aplicación
app = FastAPI(title="SkyPrice API",
                description="Hola, bienvenido a la API de **SkyPrice**. Aquí puedes predecir el precio de una propiedad utilizando tres modelos de aprendizaje automático: Random Forest, SVM y Redes Neuronales. Para predecir el precio de una propiedad, envía una solicitud POST a la ruta /predict con los datos de la propiedad. También puedes obtener información sobre los modelos y sus características en la ruta /models. ¡Diviértete!",
                version="1.0.0",
                docs_url="/openapi",
                openapi_url="/openapi.json",
                redoc_url="/redoc",
                summary="API para predecir el precio de un departamento en la CDMX",
                contact={
                    "name": "Humberto Alejandro Ortega Alcocer",
                    "email": "hortegaa1500@alumno.ipn.mx",
                    "url": "https://humbertowoody.xyz",
                },
                license_info={
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT",
                },
                terms_of_service="https://opensource.org/licenses/MIT",
                servers=[
                    {
                        "url": f"{HOSTNAME}",
                        "description": "URL de la API"
                    }
                ]
            )

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definición de una propiedad
class Property(BaseModel):
    Size_Terrain: float = Field(..., examples=[120.5], description="El tamaño del terreno en metros cuadrados")
    Size_Construction: float = Field(..., examples=[250.0], description="El tamaño de la construcción en metros cuadrados")
    Rooms: int = Field(..., examples=[3], description="El número de habitaciones")
    Bathrooms: float = Field(..., examples=[2.5], description="El número de baños")
    Parking: int = Field(..., examples=[1], description="El número de espacios de estacionamiento disponibles")
    Age: int = Field(..., examples=[5], description="La edad de la propiedad en años")
    Lat: float = Field(..., examples=[19.432608], description="La latitud de la propiedad")
    Lng: float = Field(..., examples=[-99.133209], description="La longitud de la propiedad")
    Municipality: str = Field(..., examples=["Benito Juárez"], description="El municipio donde se encuentra la propiedad")

# Definición de la respuesta del endpoint principal
class PrincipalResponse(BaseModel):
    message: str = Field(..., examples=["Bienvenido a la API de predicción inmobiliaria"], description="Mensaje de bienvenida")
    time: datetime.datetime = Field(..., examples=[datetime.datetime.now()], description="La hora actual del servidor")
    version: str = Field(..., examples=["0.1"], description="La versión de la API")
    description: str = Field(..., examples=["API para predecir el precio de una propiedad"], description="Descripción de la API")
    openapi: str = Field(..., examples=[f"{HOSTNAME}/openapi"], description="Enlace a la documentación de la API")
    redoc: str = Field(..., examples=[f"{HOSTNAME}/redoc"], description="Enlace a la documentación de la API en formato ReDoc")

# Definición de la respuesta del endpoint de predicciones
class PredictResponse(BaseModel):
    random_forest: float = Field(..., examples=[2500000.0], description="La predicción del precio de la propiedad con el modelo Random Forest")
    svm: float = Field(..., examples=[2700000.0], description="La predicción del precio de la propiedad con el modelo SVM")
    neural_network: float = Field(..., examples=[2600000.0], description="La predicción del precio de la propiedad con el modelo de Redes Neuronales")

# Definición de la respuesta del endpoint de modelos
class ModelsResponse(BaseModel):
    dataset: dict = Field(..., examples=[{"original": (1000,8),"training": {"X": (1000, 8), "y": (1000, 1)}, "testing": {"X": (250, 8), "y": (250, 1)}}], description="Información sobre los datos de entrenamiento y prueba")
    models: dict = Field(..., examples=[{"random_forest": {"mse": 1000000000.0, "ci": (900000000.0, 1100000000.0), "mae": 30000.0, "r2": 0.9, "feature_importances": [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0], "max_features": "auto", "max_depth": 10, "n_estimators": 100, "oob_score": True}, "svm": {"mse": 1500000000.0, "ci": (1400000000.0, 1600000000.0), "mae": 40000.0, "r2": 0.8, "kernel": "rbf", "C": 1.0, "epsilon": 0.1}, "neural_network": {"mse": 2000000000.0, "ci": (1900000000.0, 2100000000.0), "mae": 50000.0, "r2": 0.7, "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07}}], description="Información sobre los modelos y sus características")

# Ruta principal
@app.get("/", summary="Ruta principal", description="Ruta principal de la API de predicción inmobiliaria.", tags=["Principal"], response_description="Mensaje de bienvenida", response_model=PrincipalResponse)
async def principal():
    # Regresamos un mensaje de bienvenida, la hora actual del servidor, la fecha, la versión del API, la descripción y un link a los docs de swagger.
    return {"message": "Bienvenido a la API de SkyPrice",
            "time": datetime.datetime.now(),
            "version": "1.0.0",
            "description": "API para predecir el precio de un departamento en la Ciudad de México, si deseas ver la documentación de la API, visita el enlace de OpenAPI (Swagger UI) o ReDoc:",
            "openapi": f"{HOSTNAME}/openapi",
            "redoc": f"{HOSTNAME}/redoc"}


# Ruta para obtener las gráficas de predicciones vs valores reales
@app.get("/plots", response_class=FileResponse, summary="Obtener gráficas de predicciones vs valores reales", description="Obtener las gráficas de predicciones vs valores reales de los modelos Random Forest, SVM y Redes Neuronales.", tags=["Gráficas"], response_description="Gráficas de predicciones vs valores reales")
async def plots():
        return FileResponse('plots.png')


# Ruta para predecir el precio de una propiedad
@app.post("/predict", summary="Predecir el precio de una propiedad", description="Predecir el precio de una propiedad utilizando tres modelos de aprendizaje automático: Random Forest, SVM y Redes Neuronales.", tags=["Predicciones"], response_description="Predicciones del precio de la propiedad", response_model=PredictResponse)
async def predict(property: Property):
    # Convertir entrada a DataFrame para preprocesamiento
    input_data = [property.Municipality, property.Size_Terrain, property.Size_Construction, property.Rooms, property.Bathrooms, property.Parking, property.Age, property.Lat, property.Lng]
    input_df = pd.DataFrame([input_data], columns=['Municipality', 'Size_Terrain', 'Size_Construction', 'Rooms', 'Bathrooms', 'Parking', 'Age', 'Lat', 'Lng'])

    # Predecir con cada modelo
    rf_pred,   = rf_model.predict(input_df)
    svm_pred = svm_model.predict(input_df)[0]
    # Preprocesar para la red neuronal y luego predecir
    nn_input = preprocessor.transform(input_df)
    nn_pred = nn_model.predict(nn_input)[0][0]

    # Devolver las predicciones
    return {
        "random_forest": float(rf_pred),
        "svm": float(svm_pred),
        "neural_network": float(nn_pred)
    }

# Ruta para listar los modelos y sus características (ajuste de datos de entrenamiento, hiperparámetros, etc.)
@app.get("/models", summary="Obtener información sobre los modelos y sus características", description="Obtener información sobre los modelos de aprendizaje automático utilizados en la API, incluyendo sus características, ajuste de datos de entrenamiento, hiperparámetros, etc.", tags=["Modelos"], response_description="Información sobre los modelos y sus características", response_model=ModelsResponse)
async def models_info():
    # Regresar información sobre los modelos y sus características
    return {
        "dataset": {
            "original": df.shape,
            "training": {
                "X": X_train.shape,
                "y": y_train.shape
            },
            "testing": {
                "X": X_test.shape,
                "y": y_test.shape
            }
        },
        "models":{
            "random_forest": {
                "mse": rf_mse,
                "rmse": rf_rmse,
                "ci": rf_ci,
                "mae": rf_mae,
                "r2": rf_r2 ,
                "feature_importances": rf_model['regressor'].feature_importances_.tolist(),
                "max_features": rf_model['regressor'].max_features,
                "max_depth": rf_model['regressor'].max_depth,
                "n_estimators": rf_model['regressor'].n_estimators,
                "oob_score": rf_model['regressor'].oob_score,
            },
            "svm": {
                "mse": svm_mse,
                "rmse": svm_rmse,
                "ci": svm_ci,
                "mae": svm_mae,
                "r2": svm_r2,
                "kernel": svm_model['svr'].kernel,
                "C": svm_model['svr'].C,
                "epsilon": svm_model['svr'].epsilon
            },
            "neural_network": {
                "mse": nn_mse,
                "rmse": nn_rmse,
                "ci": nn_ci,
                "mae": nn_mae,
                "r2": nn_r2,
                "learning_rate": float(nn_model.optimizer.learning_rate.numpy()),
                "beta_1": nn_model.optimizer.beta_1,
                "beta_2": nn_model.optimizer.beta_2,
                "epsilon": nn_model.optimizer.epsilon
            }
        }
    }
