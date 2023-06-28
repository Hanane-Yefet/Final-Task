import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {
        'City': [request.form.get('City')],
        'type': [request.form.get('type')],
        'room_number': [int(request.form.get('room_number'))] if request.form.get('room_number') else [0],
        'Area': [float(request.form.get('Area'))] if request.form.get('Area') else [0.0],
        'city_area': [request.form.get('city_area')] if request.form.get('city_area') else [0.0],
        'hasElevator ': [1] if request.form.get('hasElevator') == 'on' else [0],
        'hasParking ': [1] if request.form.get('hasParking') == 'on' else [0],
        'hasBars': [1] if request.form.get('hasBars') == 'on' else [0],
        'hasStorage ': [1] if request.form.get('hasStorage') == 'on' else [0],
        'condition ': [request.form.get('condition')] if request.form.get('condition') else [0],
        'hasAirCondition': [1] if request.form.get('hasAirCondition') == 'on' else [0],
        'hasBalcony ': [1] if request.form.get('hasBalcony') == 'on' else [0],
        'hasMamad ': [1] if request.form.get('hasMamad') == 'on' else [0],
        'handicapFriendly': [1] if request.form.get('handicapFriendly') == 'on' else [0],
        'floor': [int(request.form.get('floor'))] if request.form.get('floor') else [0],
        'total_floors': [int(request.form.get('total_floors'))] if request.form.get('total_floors') else [0]
    }

    features_df = pd.DataFrame(features)

    if rf_model is not None and hasattr(rf_model, 'predict'):
        prediction = rf_model.predict(features_df)[0]
        output_text = "Predicted price: {}".format(prediction)
    else:
        output_text = "Error: Model not loaded or does not have 'predict' method"

    return render_template('index.html', prediction_text=output_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)






