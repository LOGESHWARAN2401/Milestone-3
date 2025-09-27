from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# ---- Load Dataset ----
df = pd.read_csv("2034ef0e-c373-4382-824a-3b73ddddee13.csv")  # your dataset
X = df.drop("ClassId", axis=1, errors='ignore')
y = df["ClassId"]

# Train model (just once)
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "traffic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("home"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("home"))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Normally you would preprocess image here and predict
    # For demo, just return file name
    return render_template("result.html", filename=file.filename, result="Traffic Sign Detected!")

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
