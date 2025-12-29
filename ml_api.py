from flask import Flask, request, jsonify
import pandas as pd
import random

app = Flask(__name__)

# NOTE: In the real version, we would load 'model.pkl' here.
# For now, we simulate a "Smart" decision.

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    
    predictions = []
    
    # SIMPLE LOGIC SIMULATION (Instead of ML Model)
    # If it's Loud (RMS > 0.05) OR High Motion (Motion > 0.1), KEEP IT (1)
    # Otherwise, CUT IT (0)
    for index, row in df.iterrows():
        if row['rms_volume'] > 0.05 or row['motion_score'] > 0.1:
            predictions.append(1) # Keep
        else:
            predictions.append(0) # Cut
            
    return jsonify(predictions)

if __name__ == '__main__':
    print("🤖 Model API is running on Port 5000...")
    app.run(port=5000)
