import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.externals import joblib
import mlflow
import mlflow.sklearn
import pickle

#mlflow.set_experiment("iris")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
    
if __name__ == "__main__":   
    iris = load_iris() 
    
    X = iris.data 
    y = iris.target 
    
    test_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    n_neighbors = int(sys.argv[2]) if len(sys.argv) > 1 else 3
    # Split dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 2018)
    
    with mlflow.start_run(run_name="Iris RF Experiment") as run:
        knn = KNN(n_neighbors = n_neighbors)
        # train model 
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predictions)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
    
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(knn, "knn")
        joblib.dump(knn, 'knn_model.pkl')
        #mlflow.log_artifact('knn_model.pkl', knn)
        #mlflow.models.Model.log()
