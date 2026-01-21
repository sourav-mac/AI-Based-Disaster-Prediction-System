from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and scalers
earthquake_clf = joblib.load('random_forest_classifier.pkl')
earthquake_reg = joblib.load('random_forest_regressor.pkl')
earthquake_scaler = joblib.load('scaler.pkl')
earthquake_le = joblib.load('label_encoder.pkl')

flood_clf = joblib.load('flood_classifier.pkl')
flood_scaler = joblib.load('flood_scaler.pkl')

heatwave_model = joblib.load('heatwave_model.pkl')
heatwave_scaler = joblib.load('heatwave_scaler.pkl')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# Earthquake Prediction Route
@app.route('/predict', methods=['POST', 'GET'])
def predict_earthquake():
    if request.method == 'POST':
        try:
            # Get JSON data
            data = request.json
            required_fields = ['latitude', 'longitude', 'depth']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400

            # Parse input values
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            depth = float(data['depth'])

            # Prepare input data
            input_data = pd.DataFrame({'Lat': [latitude], 'Long': [longitude], 'Depth': [depth], 'Origin Time': [0]})
            input_data[['Lat', 'Long', 'Depth', 'Origin Time']] = earthquake_scaler.transform(
                input_data[['Lat', 'Long', 'Depth', 'Origin Time']]
            )

            # Predict category and magnitude
            predicted_category_encoded = earthquake_clf.predict(input_data)
            predicted_category = earthquake_le.inverse_transform(predicted_category_encoded)
            predicted_magnitude = earthquake_reg.predict(input_data)

            return jsonify({
                'predicted_category': predicted_category[0],
                'predicted_magnitude': round(predicted_magnitude[0], 2)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('earthquake_index.html')

# Flood Prediction Route
@app.route('/predict_flood', methods=['POST', 'GET'])
def predict_flood():
    if request.method == 'POST':
        try:
            # Get JSON data
            data = request.json
            required_fields = [
                'latitude', 'longitude', 'rainfall', 'temperature',
                'humidity', 'river_discharge', 'water_level', 'elevation'
            ]
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400

            # Prepare input features
            input_data = pd.DataFrame({
                'Latitude': [float(data['latitude'])],
                'Longitude': [float(data['longitude'])],
                'Rainfall': [float(data['rainfall'])],
                'Temperature': [float(data['temperature'])],
                'Humidity': [float(data['humidity'])],
                'River Discharge': [float(data['river_discharge'])],
                'Water Level': [float(data['water_level'])],
                'Elevation': [float(data['elevation'])]
            })

            # Scale input data
            input_data_scaled = flood_scaler.transform(input_data)

            # Predict flood risk
            flood_prediction = flood_clf.predict(input_data_scaled)
            flood_result = "Flood likely" if flood_prediction[0] == 1 else "Flood unlikely"

            return jsonify({'flood_result': flood_result})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('flood_index.html')

# Heatwave Prediction Route
@app.route('/predict_heatwave', methods=['POST', 'GET'])
def predict_heatwave():
    if request.method == 'POST':
        try:
            # Get JSON data
            data = request.json
            required_fields = [
                'cloud_cover', 'precipitation_probability', 'uv_index',
                'rainfall', 'solar_radiation', 'max_temperature', 'max_humidity'
            ]
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400

            # Prepare input features
            input_features = {field: float(data[field]) for field in required_fields}
            input_df = pd.DataFrame([input_features])

            # Scale input data
            input_scaled = heatwave_scaler.transform(input_df)

            # Predict heatwave likelihood
            prediction = heatwave_model.predict(input_scaled)[0]
            result = 'Heatwave likely!' if prediction > 0.5 else 'No heatwave likely.'

            return jsonify({'heatwave_result': result})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('heatwave_index.html')

if __name__ == '__main__':
    app.run(debug=True)
