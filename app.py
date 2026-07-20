import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model Path
MODEL_PATH = "SVR_model.pkl"

# Load model using native pickle library
def load_svr_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_svr_model()

# Aesthetic UI with embedded Glassmorphism CSS & Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ SVR Revenue Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            color: var(--text-main);
            overflow-x: hidden;
        }

        /* Floating Animated Background Orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            z-index: 0;
            pointer-events: none;
            animation: floatOrb 10s ease-in-out infinite alternate;
        }
        .orb-1 { width: 350px; height: 350px; background: rgba(99, 102, 241, 0.3); top: -50px; left: -50px; }
        .orb-2 { width: 400px; height: 400px; background: rgba(236, 72, 153, 0.25); bottom: -100px; right: -50px; animation-delay: -5s; }

        @keyframes floatOrb {
            0% { transform: translateY(0) scale(1); }
            100% { transform: translateY(-30px) scale(1.08); }
        }

        .card-container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 850px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 2.5rem;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
            animation: cardAppear 0.8s ease-out;
        }

        @keyframes cardAppear {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.98rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.88rem;
            font-weight: 600;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-group input {
            width: 100%;
            padding: 0.85rem 1.1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: #a855f7;
            box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
            background: rgba(15, 23, 42, 0.85);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 16px;
            background: var(--accent-gradient);
            color: #fff;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(168, 85, 247, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.6rem;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(168, 85, 247, 0.5);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Section */
        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            text-align: center;
            display: none;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-card h3 {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }

        .result-card .value {
            font-size: 2.8rem;
            font-weight: 800;
            color: #4ade80;
            text-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
        }

        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="card-container">
        <div class="header">
            <h1>📊 SVR Revenue Predictor ⚡</h1>
            <p>Fill in the parameters below to compute the estimated revenue 💸</p>
        </div>

        <form id="predictionForm" class="grid-form">
            <div class="input-group">
                <label><i class="fa-solid fa-box-open"></i> Product Category</label>
                <input type="number" name="product_category" placeholder="e.g. 1" required>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-earth-americas"></i> Region</label>
                <input type="number" name="region" placeholder="e.g. 2" required>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-cubes"></i> Quantity</label>
                <input type="number" name="quantity" placeholder="e.g. 10" required>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-tag"></i> Unit Price ($)</label>
                <input type="number" step="0.01" name="unit_price" placeholder="e.g. 49.99" required>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-credit-card"></i> Payment Method</label>
                <input type="number" name="payment_method" placeholder="e.g. 1" required>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-truck"></i> Delivery Days</label>
                <input type="number" name="delivery_days" placeholder="e.g. 3" required>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-star"></i> Customer Rating</label>
                <input type="number" step="0.1" name="customer_rating" min="1" max="5" placeholder="1.0 - 5.0" required>
            </div>

            <button type="submit" class="btn-submit" id="submitBtn">
                <span id="btnText">🚀 Calculate Revenue</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </form>

        <div class="result-card" id="resultCard">
            <h3>Estimated Projected Revenue 🎯</h3>
            <div class="value" id="resultValue">$0.00</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultCard = document.getElementById('resultCard');
            const resultValue = document.getElementById('resultValue');

            btnText.innerText = "Calculating...";
            btnSpinner.style.display = "block";
            submitBtn.disabled = true;

            const formData = new FormData(e.target);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();

                if (data.status === 'success') {
                    resultValue.innerText = `$${parseFloat(data.prediction).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                    resultCard.style.display = 'block';
                } else {
                    alert('Prediction Error: ' + data.message);
                }
            } catch (err) {
                alert('An error occurred during estimation. Check model connection.');
            } finally {
                btnText.innerText = "🚀 Calculate Revenue";
                btnSpinner.style.display = "none";
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'SVR_model.pkl file not found or failed to load.'}), 500

    try:
        # Extract features array matching model input shape
        features = [
            float(request.form.get('product_category', 0)),
            float(request.form.get('region', 0)),
            float(request.form.get('quantity', 0)),
            float(request.form.get('unit_price', 0)),
            float(request.form.get('payment_method', 0)),
            float(request.form.get('delivery_days', 0)),
            float(request.form.get('customer_rating', 0))
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]

        return jsonify({
            'status': 'success',
            'prediction': float(prediction)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
