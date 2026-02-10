import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "laptop_price_xgb_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Feature engineering (MATCHES TRAINING)
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to training names
    df = df.rename(columns={
        "Cpu_brand": "CpuBrand",
        "Gpu_brand": "GpuBrand",
        "os": "OS",
        "Ram": "RamGB",
        "SSD": "SSD_GB",
        "HDD": "HDD_GB",
        "Ips": "IPS"
    })

    # Storage features
    df["HasSSD"] = (df["SSD_GB"] > 0).astype(int)
    df["TotalStorage_GB"] = df["SSD_GB"] + df["HDD_GB"]

    # GPU features
    df["HasDedicatedGPU"] = df["GpuBrand"].apply(
        lambda x: 0 if "intel" in x.lower() else 1
    )

    df["GpuClass"] = df["GpuBrand"].apply(
        lambda x: "Integrated" if "intel" in x.lower() else "Dedicated"
    )

    # CPU features
    df["CpuTier"] = df["CpuBrand"].apply(
        lambda x: (
            "High" if any(k in x.lower() for k in ["i7", "i9", "ryzen 7", "ryzen 9"]) else
            "Mid" if any(k in x.lower() for k in ["i5", "ryzen 5"]) else
            "Low"
        )
    )

    df["CpuPowerClass"] = df["CpuBrand"].apply(
        lambda x: "High" if "h" in x.lower() else "Normal"
    )

    df["CpuSpeedGHz"] = 2.5  # safe average used during training
    df["Height"] = 18.0      # average laptop thickness

    # Screen features
    df["PPI"] = ((1920 ** 2 + 1080 ** 2) ** 0.5) / df["Inches"]

    # OS normalization
    df["OS"] = df["OS"].apply(
        lambda x: "Windows" if "windows" in x.lower() else
                  "Mac" if "mac" in x.lower() else
                  "Linux" if "linux" in x.lower() else
                  "Other"
    )

    # Brand flags
    df["IsRazer"] = df["Company"].apply(
        lambda x: 1 if "razer" in x.lower() else 0
    )

    # Storage type
    def storage_type(row):
        if row["SSD_GB"] > 0 and row["HDD_GB"] > 0:
            return "Hybrid"
        elif row["SSD_GB"] > 0:
            return "SSD"
        else:
            return "HDD"

    df["StorageType"] = df.apply(storage_type, axis=1)

    return df


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "Company": data["Company"],
            "TypeName": data["TypeName"],
            "Ram": int(data["Ram"]),
            "Weight": float(data["Weight"]),
            "Touchscreen": int(data["Touchscreen"]),
            "Ips": int(data["Ips"]),
            "Cpu_brand": data["Cpu_brand"],
            "HDD": int(data["HDD"]),
            "SSD": int(data["SSD"]),
            "Gpu_brand": data["Gpu_brand"],
            "os": data["os"],
            "Inches": float(data["Inches"])
        }])

        # Apply feature engineering
        input_df = feature_engineering(input_df)

        # Predict log(price)
        log_price = model.predict(input_df)[0]

        # Convert back to actual price
        actual_price = np.exp(log_price)

        return jsonify({
            "predicted_price_inr": round(float(actual_price), 2)
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# -----------------------------
# Run (Render compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
