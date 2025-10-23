# app.py
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
import joblib
import pandas as pd

app = Flask(__name__)

# Liste complète des features attendues (one-hot / 0/1)
expected_features = [
        'Latitude',
        'Longitude',
        'Number_of_Casualties',
        'Number_of_Vehicles',
        'Light_Conditions_Darkness - lights lit',
        'Light_Conditions_Darkness - lights unlit',
        'Light_Conditions_Darkness - no lighting',
        'Light_Conditions_Daylight',
        'Road_Surface_Conditions_Flood over 3cm. deep',
        'Road_Surface_Conditions_Frost or ice',
        'Road_Surface_Conditions_Snow',
        'Road_Surface_Conditions_Wet or damp',
        'Road_Type_One way street',
        'Road_Type_Roundabout',
        'Road_Type_Single carriageway',
        'Road_Type_Slip road',
        'Urban_or_Rural_Area_Urban',
        'Weather_Conditions_Fine no high winds',
        'Weather_Conditions_Fog or mist',
        'Weather_Conditions_Other',
        'Weather_Conditions_Raining + high winds',
        'Weather_Conditions_Raining no high winds',
        'Weather_Conditions_Snowing + high winds',
        'Weather_Conditions_Snowing no high winds',
        'Vehicle_Type_Bus or coach (17 or more pass seats)',
        'Vehicle_Type_Car',
        'Vehicle_Type_Goods 7.5 tonnes mgw and over',
        'Vehicle_Type_Goods over 3.5t. and under 7.5t',
        'Vehicle_Type_Minibus (8 - 16 passenger seats)',
        'Vehicle_Type_Motorcycle 125cc and under',
        'Vehicle_Type_Motorcycle 50cc and under',
        'Vehicle_Type_Motorcycle over 125cc and up to 500cc',
        'Vehicle_Type_Motorcycle over 500cc',
        'Vehicle_Type_Other vehicle',
        'Vehicle_Type_Pedal cycle',
        'Vehicle_Type_Taxi/Private hire car',
        'Vehicle_Type_Van / Goods 3.5 tonnes mgw or under'
]

# Génération automatique du schéma (properties) pour Swagger / documentation
properties_dict = {feature: {"type": "integer", "example": 0} for feature in expected_features}

# Fournir le template Swagger (utilisable par Flasgger)
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "ApkoSafe Predict API",
        "description": "API de prédiction de la gravité d'accident (entrée: vecteur one-hot)",
        "version": "1.0.0"
    },
    "definitions": {
        "InputData": {
            "type": "object",
            "properties": properties_dict,
            # si tu veux forcer certains champs requis, ajoute "required": [...]
        }
    }
}

swagger = Swagger(app, template=swagger_template)

# Charger le modèle (joblib ou pickle selon comment tu l'as sauvegardé)
model = joblib.load("ApkoSafe_predict.pkl")


@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction ApkoSafe 🚦"


# utilisation de @swag_from pour documenter l'endpoint (référence au schema défini dans "definitions")
@swag_from({
    "tags": ["Prédiction"],
    "summary": "Prédire la gravité d’un accident",
    "description": "Reçoit un objet JSON avec toutes les features encodées (0 ou 1) et renvoie la prédiction.",
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "$ref": "#/definitions/InputData"
            }
        }
    ],
    "responses": {
        200: {
            "description": "Résultat de la prédiction",
            "schema": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "string"},
                    "probability": {"type": "number"}
                }
            },
            "examples": {
                "application/json": {"prediction": "Severe", "probability": 0.84}
            }
        },
        400: {
            "description": "Mauvaise requête (JSON mal formé ou champs manquants)"
        }
    }
})
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer le JSON envoyé
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Aucun JSON reçu ou JSON mal formé"}), 400

    # Construire DataFrame respectant l'ordre des colonnes attendu par le modèle
    try:
        input_df = pd.DataFrame([data], columns=expected_features).fillna(0)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la construction du DataFrame: {str(e)}"}), 400

    # Prédiction
    try:
        prediction = model.predict(input_df)[0]

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500


if __name__ == '__main__':
    # écouter toutes les interfaces (utile pour les containers / Render)
    app.run(host="0.0.0.0", port=5000, debug=True)
