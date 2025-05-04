from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pickle
import sklearn

app = Flask(__name__)

# Load Crop Recommendation Model
try:
    crop_model = joblib.load('crop_app.pkl')
except Exception as e:
    print(f"Error loading crop recommendation model: {e}")
    crop_model = None

# Load Crop Yield Prediction Model
try:
    yield_dtr = pickle.load(open('dtr.pkl','rb'))
    yield_preprocessor = pickle.load(open('preprocessor.pkl','rb'))
except Exception as e:
    print(f"Error loading yield prediction model: {e}")
    yield_dtr = None
    yield_preprocessor = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('Home_1.html')

# Crop Recommendation Routes
@app.route('/crop-recommendation')
def crop_recommendation():
    """Render the crop recommendation form"""
    return render_template('index.html')

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    """Handle crop recommendation form submission"""
    if request.method == 'POST':
        try:
            nitrogen = float(request.form.get('Nitrogen', 0))
            phosphorus = float(request.form.get('Phosphorus', 0))
            potassium = float(request.form.get('Potassium', 0))
            temperature = float(request.form.get('Temperature', 0))
            humidity = float(request.form.get('Humidity', 0))
            ph = float(request.form.get('ph', 0))
            rainfall = float(request.form.get('Rainfall', 0))

            if not (0 <= ph <= 14):
                return render_template('error.html', 
                                    message="pH must be between 0 and 14")
            
            if temperature > 100:
                return render_template('error.html',
                                    message="Temperature seems too high (max 100°C)")
            
            if humidity < 0 or humidity > 100:
                return render_template('error.html',
                                    message="Humidity must be between 0-100%")

            features = np.array([[nitrogen, phosphorus, potassium, 
                                temperature, humidity, ph, rainfall]])

            if crop_model:
                prediction = crop_model.predict(features)
                crop = str(prediction[0]).title()
                return render_template('prediction.html', prediction=crop)
            else:
                return render_template('error.html',
                                    message="Model not loaded. Please try again later.")

        except ValueError as e:
            return render_template('error.html',
                                message=f"Invalid input: {str(e)}")
        except Exception as e:
            return render_template('error.html',
                                message=f"An error occurred: {str(e)}")

# Crop Yield Prediction Routes
@app.route('/yield-prediction')
def yield_prediction():
    """Render the yield prediction form"""
    return render_template('yield_prediction.html')

@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    """Handle yield prediction form submission"""
    if request.method == 'POST':
        try:
            Year = request.form['Year']
            average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
            pesticides_tonnes = request.form['pesticides_tonnes']
            avg_temp = request.form['avg_temp']
            Area = request.form['Area']
            Item = request.form['Item']

            features = np.array([[Year, average_rain_fall_mm_per_year, 
                                pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            
            if yield_preprocessor and yield_dtr:
                transformed_features = yield_preprocessor.transform(features)
                prediction = yield_dtr.predict(transformed_features).reshape(1,-1)
                return render_template('yield_prediction.html', prediction=prediction[0][0])
            else:
                return render_template('error.html',
                                    message="Model not loaded. Please try again later.")
        except Exception as e:
            return render_template('error.html',
                                message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True,port=5003)