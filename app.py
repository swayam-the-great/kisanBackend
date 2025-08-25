import pandas as pd
import pickle
from flask import Flask, request, jsonify        
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1️⃣ Load & preprocess data
# =========================
df = pd.read_csv("All in One DataSetSwayam01cleaned.csv")

def clean_percentage(val):
    if isinstance(val, str) and "%" in val:
        return float(val.replace("%", ""))  
    try:
        return float(val)
    except:
        return None
       
for col in df.columns:
    if df[col].dtype == object:  
        if df[col].astype(str).str.contains("%").any():
            df[col] = df[col].apply(clean_percentage)

cat_cols = ["Address", "Region", "Crop"]
num_cols = [c for c in df.columns if c not in cat_cols]

df[cat_cols] = df[cat_cols].fillna("Unknown")
df[num_cols] = df[num_cols].fillna(0)

encoders = {}
for col in ["Address", "Region"]:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

X = df[["Address", "Region"]]
y = df[[
    "Nitrogen - High", "Nitrogen - Medium", "Nitrogen - Low",
    "Phosphorous - High", "Phosphorous - Medium", "Phosphorous - Low",
    "Potassium - High", "Potassium - Medium", "Potassium - Low",
    "pH - Acidic", "pH - Neutral", "pH - Alkaline"
]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained. R² Score:", model.score(X_test, y_test))

with open("soil_model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders}, f)

# =========================
# 2️⃣ Flask API
# =========================
app = Flask(__name__)
CORS(app)

@app.route("/location-info", methods=["POST"])
def location_info():
    data_input = request.get_json()
    location_input = data_input.get("location", "").strip()

    if not location_input:
        return jsonify({"error": "No location provided"}), 400

    with open("soil_model.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    encoders = saved["encoders"]

    try:
        address_enc = encoders["Address"].transform([location_input])[0]
    except:
        return jsonify({"error": f"Location '{location_input}' not found in training data"}), 404

    region_val = df[df["Address"] == address_enc]["Region"].iloc[0]
    input_features = [[address_enc, region_val]]
    predicted = model.predict(input_features)[0]

    crops = df[df["Address"] == address_enc]["Crop"].iloc[0]

    result = {
        "Address": location_input,
        "Region": encoders["Region"].inverse_transform([region_val])[0],
        "Crops": crops,
        "Attributes": {
            "Nitrogen - High": round(predicted[0], 2),
            "Nitrogen - Medium": round(predicted[1], 2),
            "Nitrogen - Low": round(predicted[2], 2),
            "Phosphorous - High": round(predicted[3], 2),
            "Phosphorous - Medium": round(predicted[4], 2),
            "Phosphorous - Low": round(predicted[5], 2),
            "Potassium - High": round(predicted[6], 2),
            "Potassium - Medium": round(predicted[7], 2),
            "Potassium - Low": round(predicted[8], 2),
            "pH - Acidic": round(predicted[9], 2),
            "pH - Neutral": round(predicted[10], 2),
            "pH - Alkaline": round(predicted[11], 2)
        }
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
