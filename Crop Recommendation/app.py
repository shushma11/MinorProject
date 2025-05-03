from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('crop_app.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('Home_1.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        """Handle form submission and return prediction"""
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

            if model:
                prediction = model.predict(features)
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
    else:
        """Render the prediction input form"""
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)