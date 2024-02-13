from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np 

app = FastAPI()

#Charger le modèle RandomForestRegressor
model = joblib.load(r"C:\Users\bacor\Documents\GitHub\Simplon-cours\API-ML\optimal_rfr_model_idf.pkl")

@app.post("/sq2_price_predictor_v1/", description="Retourne une prédiction de prix au m²")
async def sq2_price_predictor( code_postal: str , longitude = float, latitude = float ):
    input_data = np.array([[code_postal,longitude, latitude]])
    return model.predict(input_data)[0]

uvicorn.run(app)