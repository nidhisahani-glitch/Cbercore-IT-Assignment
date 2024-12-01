from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('best_lstm_model.h5')

# Load scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to preprocess input data and make predictions
def preprocess_input(data):
    print("Preprocessing input data...")
    # Assuming 'data' is a dictionary or JSON that has the necessary features
    data_df = pd.DataFrame(data)
    
    print(f"Input Data: {data_df.head()}")  # Debugging line to check input data
    
    # Scaling input features
    scaled_data = scaler.transform(data_df[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'Bollinger_Upper', 'Bollinger_Lower']])
    
    # Create sequence for LSTM input
    sequence_length = 60  # Assuming sequence length of 60
    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
    
    print(f"Processed Data Shape: {np.array(X).shape}")  # Debugging line
    return np.array(X)

@app.route('/')
def home():
    return "HELLO! I am Nidhi Sahani and I am representing my assignment to you so, Welcome to My Stock Price Prediction Flask App !!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(f"Received Data: {data}")  # Debugging line
        
        # Preprocess the input data
        input_data = preprocess_input(data)
        
        # Ensure the model is built before predicting
        if not hasattr(model, 'built') or not model.built:
            model.build(input_shape=(None, input_data.shape[1], input_data.shape[2]))
        
        # Predict stock prices using the LSTM model
        predictions = model.predict(input_data)
        print(f"Predictions: {predictions}")  # Debugging line
        
        # Return predictions as JSON response
        return jsonify(predictions.tolist())
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging line
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
