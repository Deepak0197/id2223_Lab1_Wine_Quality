import os
import modal
    
BACKFILL=False
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def generate_wine_sample(wine_type, fixed_acidity_max, fixed_acidity_min, volatile_acidity_max, volatile_acidity_min,
                         citric_acid_max, citric_acid_min, residual_sugar_max, residual_sugar_min,
                         chlorides_max, chlorides_min, free_sulfur_dioxide_max, free_sulfur_dioxide_min,
                         total_sulfur_dioxide_max, total_sulfur_dioxide_min, density_max, density_min,
                         pH_max, pH_min, sulphates_max, sulphates_min, alcohol_max, alcohol_min, quality_max, quality_min):
    """
    Returns a single row as a DataFrame representing a random sample for the wine dataset
    """
    import pandas as pd
    import random

    df = pd.DataFrame({
        "type": [wine_type],
        "fixed_acidity": [random.uniform(fixed_acidity_max, fixed_acidity_min)],
        "volatile_acidity": [random.uniform(volatile_acidity_max, volatile_acidity_min)],
        "citric_acid": [random.uniform(citric_acid_max, citric_acid_min)],
        "residual_sugar": [random.uniform(residual_sugar_max, residual_sugar_min)],
        "chlorides": [random.uniform(chlorides_max, chlorides_min)],
        "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_max, free_sulfur_dioxide_min)],
        "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_max, total_sulfur_dioxide_min)],
        "density": [random.uniform(density_max, density_min)],
        "ph": [random.uniform(pH_max, pH_min)],
        "sulphates": [random.uniform(sulphates_max, sulphates_min)],
        "alcohol": [random.uniform(alcohol_max, alcohol_min)],
        "quality": [random.randint(quality_max, quality_min)]
    })

    return df

def get_random_wine_sample():
    """
    Returns a DataFrame containing one random wine sample
    """
    import pandas as pd
    import random

    white_wine_df = generate_wine_sample(1, 7.5, 5.5, 0.6, 0.2, 0.8, 0.2, 30.0, 1.0, 0.1, 0.03, 60.0, 5.0, 300.0, 50.0, 1.01, 0.99, 3.5, 2.8, 1.0, 0.3, 14.0, 8.0, 3, 9)
    
#     white_wine_df = pd.DataFrame({
#         "type": ["white"],
#         "fixed_acidity": [random.uniform(5.5, 7.5)],
#         "volatile_acidity": [random.uniform(0.2,0.6)],
#         "citric_acid": [random.uniform(0.2,0.8)],
#         "residual_sugar": [random.uniform(1,30)],
#         "chlorides": [random.uniform(0.03,0.1)],
#         "free_sulfur_dioxide": [random.uniform(5,60)],
#         "total_sulfur_dioxide": [random.uniform(50,300)],
#         "density": [random.uniform(0.99,1.01)],
#         "ph": [random.uniform(2.8,3.5)],
#         "sulphates": [random.uniform(0.3,1)],
#         "alcohol": [random.uniform(8,14)],
#         "quality": [random.randint(3, 9)]  
#     })

    red_wine_df = generate_wine_sample(0, 8, 4.5, 0.8, 0.3, 0.8, 0.0, 15.0, 0.0, 0.5, 0.05, 50.0, 10.0, 200.0, 30.0, 1.01, 0.99, 3.8, 3.0, 2.0, 0.5, 14.0, 8.0, 3, 8)
#     red_wine_df = pd.DataFrame({
#         "type": ["red"],
#         "fixed_acidity": [random.uniform(4.5, 8)],
#         "volatile_acidity": [random.uniform(0.3,0.8)],
#         "citric_acid": [random.uniform(0,0.8)],
#         "residual_sugar": [random.uniform(0,15)],
#         "chlorides": [random.uniform(0.05,0.5)],
#         "free_sulfur_dioxide": [random.uniform(10,50)],
#         "total_sulfur_dioxide": [random.uniform(30,200)],
#         "density": [random.uniform(0.99,1.01)],
#         "ph": [random.uniform(3,3.8)],
#         "sulphates": [random.uniform(0.5,2)],
#         "alcohol": [random.uniform(8,14)],
#         "quality": [random.randint(3, 8)]  
#     })


    # randomly pick one of these 2 and return it
    pick_random = random.choice(["white", "red"])
    if pick_random == "white":
        wine_df = white_wine_df
        print("White wine sample added")
    else:
        wine_df = red_wine_df
        print("Red wine sample added")

    return wine_df

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        wine_df = pd.read_csv("winequalityN.csv")
    else:
        wine_df = get_random_wine_sample()

    wine_fg = fs.get_or_create_feature_group(
        name="wine",
        version=1,
        primary_key=["type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"],
        description="Wine quality dataset")
    wine_fg.insert(wine_df, write_options={"wait_for_job" : False})


if __name__ == "__main__":
    if LOCAL == True :
        g.local()
    else:
        with stub.run():
            f.remote()