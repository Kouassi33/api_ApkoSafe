# app.py
from flask import Flask, request, jsonify, send_file
from flasgger import Swagger, swag_from
import pickle
import pandas as pd
import folium

app = Flask(__name__)

# Initialiser le DataFrame global pour stocker les entrées
input_df = pd.DataFrame()

# Liste complète des features attendues (0/1 ou valeurs numériques)
expected_features = [
    'Latitude','Longitude','Number_of_Casualties','Number_of_Vehicles',
    'Light_Conditions_Darkness - lights lit',
    'Light_Conditions_Darkness - lights unlit',
    'Light_Conditions_Darkness - no lighting',
    'Light_Conditions_Daylight',
    'Road_Surface_Conditions_Flood over 3cm. deep',
    'Road_Surface_Conditions_Frost or ice',
    'Road_Surface_Conditions_Snow',
    'Road_Surface_Conditions_Wet or damp',
    'Road_Type_One way street','Road_Type_Roundabout','Road_Type_Single carriageway','Road_Type_Slip road',
    'Urban_or_Rural_Area_Urban',
    'Weather_Conditions_Fine no high winds','Weather_Conditions_Fog or mist','Weather_Conditions_Other',
    'Weather_Conditions_Raining + high winds','Weather_Conditions_Raining no high winds',
    'Weather_Conditions_Snowing + high winds','Weather_Conditions_Snowing no high winds',
    'Vehicle_Type_Bus or coach (17 or more pass seats)','Vehicle_Type_Car',
    'Vehicle_Type_Goods 7.5 tonnes mgw and over','Vehicle_Type_Goods over 3.5t. and under 7.5t',
    'Vehicle_Type_Minibus (8 - 16 passenger seats)',
    'Vehicle_Type_Motorcycle 125cc and under','Vehicle_Type_Motorcycle 50cc and under',
    'Vehicle_Type_Motorcycle over 125cc and up to 500cc','Vehicle_Type_Motorcycle over 500cc',
    'Vehicle_Type_Other vehicle','Vehicle_Type_Pedal cycle','Vehicle_Type_Taxi/Private hire car',
    'Vehicle_Type_Van / Goods 3.5 tonnes mgw or under'
]

# Charger le modèle
with open("ApkoSafe_predict.pkl", "rb") as f:
    model = pickle.load(f)

# Classes et couleurs
classes = {0: "faible", 1: "grave", 2: "mortel"}
colors = {0: "green", 1: "orange", 2: "red"}

# Swagger
properties_dict = {feature: {"type": "number", "example": 0} for feature in expected_features}
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "ApkoSafe Predict API",
        "description": "API de prédiction de la gravité d'accident et visualisation sur carte",
        "version": "1.0.0"
    },
    "definitions": {
        "InputData": {
            "type": "object",
            "properties": properties_dict,
            "required": expected_features
        }
    }
}
swagger = Swagger(app, template=swagger_template)


@app.route('/')
def home():
    return "Bienvenue sur l'API ApkoSafe 🚦"


@app.route('/predict', methods=['POST'])
@swag_from({
    "tags": ["Prédiction"],
    "summary": "Prédire la gravité d’un accident et générer une carte",
    "parameters": [{
        "name": "body",
        "in": "body",
        "required": True,
        "schema": {"$ref": "#/definitions/InputData"}
    }],
    "responses": {
        200: {
            "description": "Résultat de la prédiction et carte générée",
            "schema": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "string"},
                    "probabilities": {"type": "object"},
                    "carte_file": {"type": "string"}
                }
            }
        },
        400: {"description": "JSON mal formé ou champs manquants"}
    }
})
def predict():
    global input_df
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Aucun JSON reçu ou JSON mal formé"}), 400

    try:
        # Ajouter la nouvelle entrée dans le DataFrame global
        new_row = pd.DataFrame([data], columns=expected_features).fillna(0)
        input_df = pd.concat([input_df, new_row], ignore_index=True)

        # Prédiction
        pred = model.predict(new_row)[0]
        proba_vals = model.predict_proba(new_row)[0]
        proba_dict = {classes[i]: round(p,2) for i,p in enumerate(proba_vals)}

        # Génération de la carte
        couleur = colors[pred]
        carte = folium.Map(location=[new_row['Latitude'].iloc[0], new_row['Longitude'].iloc[0]], zoom_start=10)
        for idx, row in input_df.iterrows():
            folium.Circle(
                location=[row['Latitude'], row['Longitude']],
                radius=300,
                color=colors[model.predict(pd.DataFrame([row], columns=expected_features))[0]],
                fill=True,
                fill_color=colors[model.predict(pd.DataFrame([row], columns=expected_features))[0]],
                fill_opacity=0.3,
                popup=f"risk: {classes[model.predict(pd.DataFrame([row], columns=expected_features))[0]]}"
            ).add_to(carte)
        carte_file = 'carte_depuis_api.html'
        carte.save(carte_file)

        return jsonify({
            "prediction": classes[pred],
            "probabilities": proba_dict,
            "carte_file": carte_file
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_map', methods=['GET'])
def get_map():
    """Permet de télécharger la carte générée"""
    try:
        return send_file('carte_depuis_api.html')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
