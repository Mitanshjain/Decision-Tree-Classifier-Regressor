from flask import Flask, request, jsonify, Response
import joblib
import numpy as np

app = Flask(__name__)

# Load the model once
model = joblib.load('dt_model.pkl')


# How website works :-
# website ---> button ---> /about(URL)  ---> def about(design + code) ---> Display


@app.route('/predict',methods=['POST'])
def predict() -> Response:
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1,-1)
        prediction = model.predict(input_data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)