"""
API minima para el pipeline exploratorio de mortalidad intrahospitalaria
en endocarditis (Taller 2 - Analisis de Datos I).

Carga el pipeline entrenado (Elastic Net + preprocesamiento) directamente
desde modelo_endocarditis_exploratorio.joblib y expone un endpoint /predict
que devuelve la puntuacion de riesgo real calculada por el modelo, no una
aproximacion escrita a mano.

Uso exploratorio y academico. No debe usarse para decisiones clinicas.
"""

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = "modelo_endocarditis_exploratorio.joblib"

app = FastAPI(
    title="Riesgo exploratorio - Endocarditis",
    description=(
        "Puntuacion de riesgo exploratoria de mortalidad intrahospitalaria "
        "en endocarditis, a partir del pipeline entrenado en el Taller 2. "
        "Uso academico; no representa una probabilidad calibrada ni debe "
        "usarse para decisiones clinicas."
    ),
    version="1.0.0",
)

# CORS abierto: el frontend vive en un dominio distinto (Artifact publicado).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

paquete = joblib.load(MODEL_PATH)
pipeline = paquete["pipeline"]
variables_entrada = paquete["variables_entrada"]
umbral_riesgo_elevado = paquete["umbral_riesgo_elevado"]
etiquetas = paquete["etiquetas"]
uso_previsto = paquete["uso_previsto"]
nota_porcentaje = paquete["nota_porcentaje"]


class PacienteInput(BaseModel):
    Edad: float = Field(..., ge=0, le=120, description="Edad en anios")
    Falla_cardiaca: Literal["No", "Si"] = Field(..., alias="Falla cardiaca")
    Enfermedad_renal: Literal[
        "No", "Si, con dialisis peritoneal", "Si, con hemodialisis", "Si, sin dialisis"
    ] = Field(..., alias="Enfermedad renal")
    Tipo_de_endocarditis: Literal[
        "Nativa", "Protesica", "Asociada a cateter", "Asociada a Dispositvo cardiaco"
    ] = Field(..., alias="Tipo de endocarditis")
    Staphylococcus_aureus: Literal[
        "No", "Si, meticilino sensible", "Si, meticilino resistente"
    ] = Field(..., alias="Staphylococcus aureus")
    Neutrofilos: float = Field(..., ge=0, description="Neutrofilos absolutos (cel/uL)")
    Creatinina: float = Field(..., ge=0, description="Creatinina serica (mg/dL)")
    Plaquetas: float = Field(..., ge=0, description="Plaquetas (cel/uL)")
    Hemoglobina: float = Field(..., ge=0, description="Hemoglobina (g/dL)")
    Fiebre: Literal["No", "Si"]
    Sintomas_neurologicos: Literal["No", "Si"] = Field(..., alias="Sintomas neurologicos")
    Aneurisma_micotico: Literal["No", "Si"] = Field(..., alias="Aneurisma micotico")

    class Config:
        populate_by_name = True


class PrediccionOutput(BaseModel):
    puntuacion_riesgo_pct: float
    categoria: str
    umbral_riesgo_elevado: float
    uso_previsto: str
    nota_porcentaje: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "modelo": "Elastic Net - mortalidad intrahospitalaria en endocarditis (exploratorio)",
        "uso_previsto": uso_previsto,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PrediccionOutput)
def predict(paciente: PacienteInput):
    datos = paciente.model_dump(by_alias=True)
    fila = {var: datos[var] for var in variables_entrada}
    X = pd.DataFrame([fila], columns=variables_entrada)

    try:
        proba = float(pipeline.predict_proba(X)[0, 1])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Error al calcular la prediccion: {exc}") from exc

    categoria = (
        etiquetas["riesgo_elevado"]
        if proba >= umbral_riesgo_elevado
        else etiquetas["riesgo_no_elevado"]
    )

    return PrediccionOutput(
        puntuacion_riesgo_pct=round(proba * 100, 1),
        categoria=categoria,
        umbral_riesgo_elevado=umbral_riesgo_elevado,
        uso_previsto=uso_previsto,
        nota_porcentaje=nota_porcentaje,
    )
