import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('insurance_charges_pipeline.joblib')

def predict_insurance_charges(age, sex, bmi, children, smoker, region):
    """
    Predict insurance charges based on input features.
    
    Parameters:
    -----------
    age : int
        Age of the person
    sex : str
        Gender of the person ('male' or 'female')
    bmi : float
        Body Mass Index
    children : int
        Number of children/dependents
    smoker : str
        Smoking status ('yes' or 'no')
    region : str
        Geographic region ('northeast', 'northwest', 'southeast', 'southwest')
        
    Returns:
    --------
    float
        Predicted insurance charges
    """
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Make prediction using the loaded pipeline
    prediction = pipeline.predict(data)[0]
    
    return prediction

def batch_predict_insurance_charges(data_list):
    """
    Predict insurance charges for multiple individuals.
    
    Parameters:
    -----------
    data_list : list of dict
        List of dictionaries containing features for each individual
        
    Returns:
    --------
    list
        List of predicted insurance charges
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    
    # Make predictions
    predictions = pipeline.predict(df)
    
    return predictions
