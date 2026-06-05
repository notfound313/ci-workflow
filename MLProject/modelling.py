import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
import os
import numpy as np
import warnings
import sys


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    df = pd.read_csv(file_path)

    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    
    scaler = StandardScaler()
    X_train_stand = scaler.fit_transform(X_train)
    X_test_stand = scaler.transform(X_test)
    
    input_example = X_train_stand[0:5]
    
    params = {
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'random_state': 42,
        'C': 10,
        'gamma': 0.001
    }
    svm_model = SVC(**params)   
   

    with mlflow.start_run():        
        svm_model.fit(X_train_stand, y_train)

        
        predicted_qualities = svm_model.predict(X_test_stand)
        
        mlflow.log_params(params)
        
        mlflow.sklearn.log_model(
            sk_model=svm_model,
            artifact_path="model",
            input_example=input_example
        )

        
        accuracy = svm_model.score(X_test_stand, y_test)
        mlflow.log_metric("accuracy", accuracy)

   