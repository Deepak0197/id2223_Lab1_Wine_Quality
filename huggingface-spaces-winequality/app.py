import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import numpy as np


project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine_quality(type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                 free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    print("Calling function")
    df = pd.DataFrame([[type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]],
                      columns=["type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                               "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                               "ph", "sulphates", "alcohol"])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    print(res)
    try:
        quality_url = f"https://raw.githubusercontent.com/Deepak0197/id2223_Lab1_Wine_Quality/main/Wine_dataset/{int(np.round(res[0]))}.png"
        response = requests.get(quality_url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        img = Image.open(response.raw)
        return img
    except Exception as e:
        print(f"Error downloading or opening the image: {e}")
        return None

demo = gr.Interface(
    fn=wine_quality,
    title="Wine Quality Predictive Analytics",
    description="Experiment with Various wine parameters to predict What is the quality of the wine.",
    allow_flagging="never",
    inputs=[
        gr.Number(label="type (0-Red or 1-White)"),
        gr.Number(label="fixed_acidity (Between 3.8 - 14.2)"),
        gr.Number(label="volatile_acidity (Between 0.1 - 1.02)"),
        gr.Number(label="citric_acid (Between 0 - 1.66)"),
        gr.Number(label="residual_sugar (Between 0.7 - 19.5)"),
        gr.Number(label="chlorides (Between 0.014 - 0.611)"),
        gr.Number(label="free_sulfur_dioxide (Between 2 - 138.5)"),
        gr.Number(label="total_sulfur_dioxide (Between 7 - 366.5)"),
        gr.Number(label="density (Between 0.98746 - 1.00289)"),
        gr.Number(label="ph (Between 2.77 - 3.85)"),
        gr.Number(label="sulphates (Between 0.26 - 1.36)"),
        gr.Number(label="alcohol (Between 8 - 14.2)"),
    ],
    outputs=gr.Image(type="pil")
)

demo.launch(debug=True, share=True)


