from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model
model = load_model("skincare_haircare_recommendation_model.h5")

# Load the dataset to get the original labels
skincare_haircare_data = pd.read_csv("skincare_haircare_data.csv")

def make_recommendation(skin_type, hair_type):
    # Transform input data similar to training data
    le_skin = LabelEncoder()
    le_hair = LabelEncoder()

    # Fit the label encoders with the original data to ensure consistent encoding
    le_skin.fit(skincare_haircare_data['Skintype'])
    le_hair.fit(skincare_haircare_data['Hairtype'])

    skin_encoded = le_skin.transform([skin_type])
    hair_encoded = le_hair.transform([hair_type])
    user_input = np.array([skin_encoded[0], hair_encoded[0]]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(user_input)
    
    # Decode the predicted product index
    product_index = np.argmax(prediction)
    
    # Retrieve the product name using the index from the dataset
    decoded_product = skincare_haircare_data.loc[product_index, 'Product']
    
    return decoded_product

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    skin_type = data.get('skin_type')
    hair_type = data.get('hair_type')

    if skin_type is None or hair_type is None:
        return jsonify({'error': 'Please provide both skin_type and hair_type in the request.'}), 400

    recommendation = make_recommendation(skin_type, hair_type)
    return jsonify({'recommended_product': recommendation})

if __name__ == '__main__':
    app.run(debug=True)
