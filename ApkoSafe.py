# app.py
from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import pandas as pd

app = Flask(__name__)
swagger = Swagger(app)

# Charger le modèle
model = joblib.load("ApkoSafe_predict.pkl")

# Liste complète des features
expected_features = [
    "Light_Conditions_Darkness - lights lit",
    "Light_Conditions_Darkness - lights unlit",
    "Light_Conditions_Darkness - no lighting",
    "Light_Conditions_Daylight",
    "Road_Surface_Conditions_Flood over 3cm. deep",
    "Road_Surface_Conditions_Frost or ice",
    "Road_Surface_Conditions_Snow",
    "Road_Surface_Conditions_Wet or damp",
    "Road_Type_One way street",
    "Road_Type_Roundabout",
    "Road_Type_Single carriageway",
    "Road_Type_Slip road",
    "Urban_or_Rural_Area_Unallocated",
    "Urban_or_Rural_Area_Urban",
    "Weather_Conditions_Fine no high winds",
    "Weather_Conditions_Fog or mist",
    "Weather_Conditions_Other",
    "Weather_Conditions_Raining + high winds",
    "Weather_Conditions_Raining no high winds",
    "Weather_Conditions_Snowing + high winds",
    "Weather_Conditions_Snowing no high winds",
    "Vehicle_Type_Bus or coach (17 or more pass seats)",
    "Vehicle_Type_Car",
    "Vehicle_Type_Data missing or out of range",
    "Vehicle_Type_Goods 7.5 tonnes mgw and over",
    "Vehicle_Type_Goods over 3.5t. and under 7.5t",
    "Vehicle_Type_Minibus (8 - 16 passenger seats)",
    "Vehicle_Type_Motorcycle 125cc and under",
    "Vehicle_Type_Motorcycle 50cc and under",
    "Vehicle_Type_Motorcycle over 125cc and up to 500cc",
    "Vehicle_Type_Other vehicle",
    "Vehicle_Type_Pedal cycle",
    "Vehicle_Type_Ridden horse",
    "Vehicle_Type_Taxi/Private hire car",
    "Vehicle_Type_Van / Goods 3.5 tonnes mgw or under",
    "Vehicle_Type_Motorcycle over 500cc"
]


@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction ApkoSafe"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédiction du niveau de gravité d’un accident
    ---
    tags:
      - Prédiction
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - name: input_data
        in: body
        required: true
        schema:
          type: object
          properties:
            {%- for feature in features %}
            "{{ feature }}":
              type: number
              example: 0
            {%- endfor %}
    responses:
      200:
        description: Résultat de la prédiction
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: "Severe"
    """
    data = request.get_json()

    # Vérification des colonnes
    input_df = pd.DataFrame([data], columns=expected_features).fillna(0)

    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.jinja_env.globals.update(features=expected_features)  # pour Swagger dynamic
    app.run(debug=True)
