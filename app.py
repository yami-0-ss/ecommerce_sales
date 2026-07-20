import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the SVR Model
MODEL_PATH = "discount_SVR.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Running in simulation/demo mode.")

# HTML Template with Embedded Modern Dark-glass UI and CSS Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discount Predictor SVR - Dashboard</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent: #22d3ee;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.12);
            --radius-lg: 20px;
            --radius-md: 12px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Animated Ambient Background Orbs */
        .ambient-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }

        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.4;
            animation: float 18s infinite ease-in-out alternate;
        }

        .orb-1 {
            width: 400px;
            height: 400px;
            background: #6366f1;
            top: -100px;
            left: -100px;
        }

        .orb-2 {
            width: 450px;
            height: 450px;
            background: #a855f7;
            bottom: -150px;
            right: -100px;
            animation-delay: -5s;
        }

        .orb-3 {
            width: 300px;
            height: 300px;
            background: #22d3ee;
            top: 40%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation-delay: -10s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(60px, 40px) scale(1.1); }
            100% { transform: translate(-40px, 80px) scale(0.95); }
        }

        /* Glassmorphism Card Container */
        .container {
            position: relative;
            z-index: 10;
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-lg);
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(99, 102, 241, 0.15);
            color: var(--accent);
            border: 1px solid rgba(34, 211, 238, 0.3);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Grid Layout */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-group label i {
            color: var(--accent);
            font-size: 0.9rem;
        }

        .input-control {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
            background: rgba(15, 23, 42, 0.8);
        }

        select.input-control {
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394a3b8'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1rem center;
            background-size: 1.2rem;
            cursor: pointer;
        }

        /* Submit Button */
        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--primary-gradient);
            color: #fff;
            border: none;
            border-radius: var(--radius-md);
            padding: 1rem;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
            filter: brightness(1.05);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Popup Card */
        .result-card {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(34, 211, 238, 0.1) 100%);
            border: 1px solid rgba(99, 102, 241, 0.4);
            text-align: center;
            animation: pulseIn 0.5s ease-out forwards;
        }

        @keyframes pulseIn {
            0% { opacity: 0; transform: scale(0.95); }
            100% { opacity: 1; transform: scale(1); }
        }

        .result-card h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #fff;
            text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 600px) {
            .container { padding: 1.5rem; }
            .header h1 { font-size: 1.75rem; }
        }
    </style>
</head>
<body>

    <div class="ambient-bg">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>

    <div class="container">
        <div class="header">
            <div class="badge"><i class="fa-solid fa-brain"></i> SVR Machine Learning Model</div>
            <h1>Discount Rate Predictor</h1>
            <p>Enter transactional & customer parameters to predict optimal discount rates in real-time.</p>
        </div>

        <form id="predictionForm" class="form-grid">
            <div class="input-group">
                <label><i class="fa-solid fa-id-badge"></i> Customer ID</label>
                <input type="number" class="input-control" name="customer_id" placeholder="e.g. 1024" required value="101">
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-layer-group"></i> Product Category</label>
                <select class="input-control" name="product_category" required>
                    <option value="0">Electronics</option>
                    <option value="1">Fashion</option>
                    <option value="2">Home & Garden</option>
                    <option value="3">Beauty & Care</option>
                </select>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-earth-americas"></i> Region</label>
                <select class="input-control" name="region" required>
                    <option value="0">North America</option>
                    <option value="1">Europe</option>
                    <option value="2">Asia Pacific</option>
                    <option value="3">Latin America</option>
                </select>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-boxes-stacked"></i> Quantity</label>
                <input type="number" class="input-control" name="quantity" min="1" placeholder="e.g. 5" required value="3">
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-tag"></i> Unit Price ($)</label>
                <input type="number" step="0.01" class="input-control" name="unit_price" placeholder="e.g. 49.99" required value="85.50">
            </div>

            <div class="input-group">
                <label><i class="fa-credit-card fa-solid"></i> Payment Method</label>
                <select class="input-control" name="payment_method" required>
                    <option value="0">Credit Card</option>
                    <option value="1">PayPal</option>
                    <option value="2">Debit Card</option>
                    <option value="3">Bank Transfer</option>
                </select>
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-truck-fast"></i> Delivery Days</label>
                <input type="number" class="input-control" name="delivery_days" min="1" placeholder="e.g. 3" required value="2">
            </div>

            <div class="input-group">
                <label><i class="fa-solid fa-star"></i> Customer Rating (1-5)</label>
                <input type="number" step="0.1" min="1" max="5" class="input-control" name="customer_rating" placeholder="e.g. 4.5" required value="4.8">
            </div>

            <div class="input-group" style="grid-column: 1 / -1;">
                <label><i class="fa-solid fa-dollar-sign"></i> Total Revenue ($)</label>
                <input type="number" step="0.01" class="input-control" name="revenue" placeholder="e.g. 256.50" required value="256.50">
            </div>

            <button type="submit" class="btn-submit">
                <span id="btnText">Calculate Predicted Discount</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </form>

        <div class="result-card" id="resultCard">
            <h3>Predicted Discount Percentage</h3>
            <div class="result-value" id="resultValue">0.00%</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultCard = document.getElementById('resultCard');
            const resultValue = document.getElementById('resultValue');

            btnText.innerText = "Analyzing SVR Model...";
            btnSpinner.style.display = "block";
            resultCard.style.display = "none";

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    resultValue.innerText = `${result.predicted_discount.toFixed(2)}%`;
                    resultCard.style.display = "block";
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (err) {
                alert('Request failed. Check server connection.');
            } finally {
                btnText.innerText = "Calculate Predicted Discount";
                btnSpinner.style.display = "none";
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extracted 9 features:
        # ['customer_id', 'product_category', 'region', 'quantity', 'unit_price', 'payment_method', 'delivery_days', 'customer_rating', 'revenue']
        features = [
            float(data.get("customer_id", 0)),
            float(data.get("product_category", 0)),
            float(data.get("region", 0)),
            float(data.get("quantity", 0)),
            float(data.get("unit_price", 0)),
            float(data.get("payment_method", 0)),
            float(data.get("delivery_days", 0)),
            float(data.get("customer_rating", 0)),
            float(data.get("revenue", 0))
        ]
        
        features_array = np.array([features])
        
        if model is not None:
            prediction = model.predict(features_array)[0]
        else:
            prediction = (features[3] * features[4] * 0.05) + (features[7] * 2.0)
            
        return jsonify({"predicted_discount": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
