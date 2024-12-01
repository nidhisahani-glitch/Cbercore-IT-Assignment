# Cbercore-IT-Assignment
📂 Stock_Price_Prediction
├── 📄 LSE Dataset.csv
├── 📄 NYSE Dataset.csv
├── 📄 Stock_Prediction.ipynb       # Jupyter Notebook for the project
├── 📄 flask_app.py                 # Flask application
├── 📄 README.md                    # Project documentation
└── 📂 models                       # Directory to save trained models
Run the Jupyter Notebook: Preprocess the data and train the model using the notebook Stock_Prediction.ipynb.
Hyperparameter Tuning: Execute the hyperparameter tuning code using RandomizedSearchCV.
Launch the Flask App: Start the Flask server to make predictions:
python flask_app.py
Access the API at http://127.0.0.1:5000/predict.
Make Predictions: Send a POST request to the API endpoint with input features (e.g., normalized technical indicators).

Key Libraries
TensorFlow/Keras: For building and training LSTM models.
Scikit-learn: For preprocessing, metrics, and hyperparameter tuning.
Scikeras: Wrapper for Keras models compatible with Scikit-learn utilities.
Flask: For deploying the model as a REST API.
