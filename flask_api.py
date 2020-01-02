import numpy as np 
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib
import flask

app = Flask(__name__)
api = Api(app)

model = None
parser = reqparse.RequestParser()
parser.add_argument('sepal_length', type=float, required=True)
parser.add_argument('sepal_width', type=float, required=True)
parser.add_argument('petal_length', type=float, required=True)
parser.add_argument('petal_width', type=float, required=True)


def load_model(ml_model):
    global model
    model = joblib.load(ml_model)

#def input_check(input_parameters):
#    try:
#        parameters = np.array(input_parameters)
#        n,x = parameters.shape
#    except ValueError:
#        message = "Value error occured: Shape of the input is not matching"
#        check = False
#    else:
#        message = "Success"
#        check = True
#    
#    return parameters, message, check 


class Predict(Resource):
    def get(self):
        result = {}
        result['success'] = False
        
        #if flask.request.files.get("input_params"):
        args = parser.parse_args()
        #input_params, message, check = args['input_params']
        sepal_length, sepal_width = args['sepal_length'], args['sepal_width'], 
        petal_length, petal_width = args['petal_length'], args['petal_width']
        #if check:
        input_params = [[sepal_length, sepal_width, petal_length, petal_width]]
        predict = model.predict(input_params)
        prediction = str(list(predict)[0])
        if prediction == 0:
            result['prediction'] = "setosa"
        elif prediction == 1:
            result['prediction'] = "versicolor"
        else:
            result['prediction'] = "virginica"
        # str(list(predict)[0])
        result['success'] = True
        result['message'] = "success"
        #else:
        #    result['prediction'] = []
        #    result['message'] = message
        
        return flask.jsonify(result)

api.add_resource(Predict, '/')

if __name__ =='__main__':
    load_model('knn_iris_model.pkl')
    app.run(debug = True)
