import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine_quality(type,fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[type,fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]], 
                      columns=["type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol"])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    quality_url = "https://raw.githubusercontent.com/Deepak0197/id2223_Lab1_Wine_Quality/main/Wine_dataset/" + res[0] + ".png"
    img = Image.open(requests.get(quality_url, stream=True).raw)            
    return img    
 
        
demo = gr.Interface(
    fn=wine_quality,
    title="Wine Quality Predictive Analytics",
    description="Experiment with Various wine parameters to predict What is the quality of the wine.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=0, label="type (0-Red or 1-White)"),
        gr.inputs.Number(default=7.0, label="fixed_acidity (Between 3.8 - 14.2)"),
        gr.inputs.Number(default=0.3, label="volatile_acidity (Between 0.1 - 1.02)"),
        gr.inputs.Number(default=0.3, label="citric_acid (Between 0 - 1.66)"),
        gr.inputs.Number(default=4.7, label="residual_sugar (Between 0.7 - 19.5)"),
        gr.inputs.Number(default=0.05, label="chlorides (Between 0.014 - 0.611)"),
        gr.inputs.Number(default=29.7, label="free_sulfur_dioxide (Between 2 - 138.5)"),
        gr.inputs.Number(default=113.01, label="total_sulfur_dioxide (Between 7 - 366.5)"),
        gr.inputs.Number(default=0.99, label="density (Between 0.98746 - 1.00289)"),
        gr.inputs.Number(default=3.22, label="ph (Between 2.77 - 3.85)"),
        gr.inputs.Number(default=0.53, label="sulphates (Between 0.26 - 1.36)"),
        gr.inputs.Number(default=10.57, label="alcohol (Between 8 - 14.2)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

