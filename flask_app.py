import git
import matplotlib.pyplot as plt
import shap
import joblib
import pandas as pd
import os
from flask import Flask, jsonify, request, send_file

# commentaire de test
# commentaire de test 2
# commentaire de test 3

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

# Load the model
model = joblib.load('/home/patriciaxdubray/api/model.pkl')

# Load the MinMaxScaler from mms.pkl
mms = joblib.load('/home/patriciaxdubray/api/mms.pkl')

# Load the explainer SHAP from shap.pkl
explainer = joblib.load('/home/patriciaxdubray/api/shap.pkl')


# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the data sent as JSON
    data = request.get_json()

    # Create a DataFrame from the data
    df = pd.DataFrame([data], index=[0])

    # Normalize the input data using MinMaxScaler
    df_normalized = pd.DataFrame(mms.transform(df), columns=df.columns)

    # Make predictions
    proba = model.predict_proba(df_normalized)
    class_0_proba = proba[0][0]
    class_1_proba = proba[0][1]

    class_0_percentage = round(class_0_proba * 100, 1)
    class_1_percentage = round(class_1_proba * 100, 1)

    class_0 = f"Classe 0 : {class_0_percentage}% de probabilité que le remboursement soit ok"
    class_1 = f"Classe 1 : {class_1_percentage}% de probabilité d'avoir des problèmes de remboursement"

    # Generate SHAP visualization for the observation
    shap_values = explainer.shap_values(df_normalized.iloc[[0]])

    # Summary plot
    shap.summary_plot(shap_values, df_normalized.iloc[[0]], plot_type="bar")

    # Save the summary plot to a file
    summary_plot_path = '/home/patriciaxdubray/api/summary_plot.png'
    plt.savefig(summary_plot_path)
    plt.clf()  # Clear the plot

    result = {
        "class_0": class_0,
        "class_1": class_1,
        "summary_plot_path": summary_plot_path
    }

    return jsonify(result)


# Route for downloading the summary plot
@app.route('/summary_plot', methods=['GET'])
def download_plot():
    return send_file('/home/patriciaxdubray/api/summary_plot.png', as_attachment=True)


# Route for the home page
@app.route('/')
def home():
    return "API dashboard prêt à dépenser"


@app.route('/update_server', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('/home/patriciaxdubray/api')
        origin = repo.remotes.origin
        origin.pull()
        return 'Updated PythonAnywhere successfully', 200
    else:
        return 'Wrong event type', 400


if __name__ == '__main__':
    # Display a message when the URL is launched
    print("API dashboard prêt à dépenser")
