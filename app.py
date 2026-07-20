import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the SVR model from the pickle file
MODEL_PATH = 'SVR_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    print(f"Warning: {MODEL_PATH} not found. Ensure it is placed in the root directory.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVR Model Predictor</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0f172a);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        body {
            min-height: 100vh;
            background: var(--bg-gradient);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: var(--text-main);
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h2 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 28px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        input[type="text"] {
            width: 100%;
            padding: 14px 18px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input[type="text"]:focus {
            border-color: var(--accent-color);
            background: rgba(255, 255, 255, 0.12);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        }

        button {
            width: 100%;
            padding: 14px;
            background: var(--accent-color);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
            margin-top: 10px;
        }

        button:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        button:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 28px;
            padding: 20px;
            border-radius: 12px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-align: center;
            animation: pulseIn 0.5s ease;
        }

        @keyframes pulseIn {
            0% { transform: scale(0.95); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-box h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            color: #a5b4fc;
            margin-bottom: 6px;
        }

        .result-box p {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
        }

        .error-box {
            margin-top: 20px;
            padding: 14px;
            border-radius: 12px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            font-size: 0.9rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>SVR Model Inference</h2>
        <p class="subtitle">Enter comma-separated features to run prediction</p>
        
        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="features">Input Features</label>
                <input type="text" id="features" name="features" placeholder="e.g., 2.5, 3.1, 0.4" required value="{{ input_val or '' }}">
            </div>
            <button type="submit">Predict Value</button>
        </form>

        {% if result is not none %}
        <div class="result-box">
            <h3>Predicted Result</h3>
            <p>{{ result }}</p>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-box">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, result=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, result=None, error="Model file SVR_model.pkl not found on server.")

    raw_input = request.form.get('features', '')
    try:
        # Convert comma-separated string to float list
        features = [float(x.strip()) for x in raw_input.split(',') if x.strip()]
        
        if not features:
            raise ValueError("No valid numbers entered.")

        # Reshape for single sample prediction
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        
        # Format prediction float nicely
        formatted_pred = round(float(prediction), 4)

        return render_template_string(HTML_TEMPLATE, result=formatted_pred, error=None, input_val=raw_input)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=None, error=f"Invalid Input: {str(e)}", input_val=raw_input)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
