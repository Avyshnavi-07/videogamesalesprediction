from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import gdown
import os

app = Flask(__name__)

# ===== Google Drive File IDs =====
MODEL_ID = "1xTiuPkNDS8ypPuSgt5U705FadAL2gypo"          # replace with your model file ID
ENCODERS_ID = "1RtnA1kOkJ6e4kuBBw_GVMMSeiUmiGhVB"       # replace with your encoders file ID
FEATURES_ID = "1tw-tBEB1mH_KPC68jj6nYnJyn6ivBOBv"      # replace with your feature names file ID

# ===== Local Paths =====
MODEL_PATH = "rf_model.pkl"
ENCODERS_PATH = "rf_encoders.pkl"
FEATURES_PATH = "rf_feature_names.pkl"

# ===== Helper function to download file if missing =====
def download_from_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading {dest_path} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

# ===== Ensure model files are available =====
download_from_drive(MODEL_ID, MODEL_PATH)
download_from_drive(ENCODERS_ID, ENCODERS_PATH)
download_from_drive(FEATURES_ID, FEATURES_PATH)

# ===== Load trained model, encoders & feature names =====
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, 'rb') as f:
    encoders = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ----- Get form inputs -----
        platform = request.form['platform']
        genre = request.form['genre']
        publisher = request.form['publisher']
        year = float(request.form['year'])
        na_sales = float(request.form['na_sales'])
        eu_sales = float(request.form['eu_sales'])
        jp_sales = float(request.form['jp_sales'])
        other_sales = float(request.form['other_sales'])

        # ----- Create input DataFrame -----
        input_df = pd.DataFrame([{
            'Platform': platform,
            'Genre': genre,
            'Publisher': publisher,
            'Year': year,
            'NA_Sales': na_sales,
            'EU_Sales': eu_sales,
            'JP_Sales': jp_sales,
            'Other_Sales': other_sales
        }])

        # ----- Apply encoding safely -----
        for col, encoder in encoders.items():
            if col in input_df.columns:
                value = input_df.at[0, col]
                if value in encoder.classes_:
                    input_df.at[0, col] = encoder.transform([value])[0]
                else:
                    print(f"⚠️ Unseen label '{value}' in column '{col}'")
                    input_df.at[0, col] = 0

        # ----- Reorder columns -----
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # ----- Predict -----
        pred = model.predict(input_df)[0]
        pred = round(pred, 3)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Global Sales: {pred} million units'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
