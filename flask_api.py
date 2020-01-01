import numpy as np 
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)

model = None
parser = reqparse.RequestParser()
parser.add_argument('input_params')


def load_model(ml_model):
    global model
    model = joblib.load(ml_model)

def input_check(input_parameters):
    try:
        parameters = np.array(input_parameters)
        n,x = parameters.shape
    except ValueError:
        message = "Value error occured: Shape of the input is not matching"
        check = False
    elif x != 4:
        message = "Expected 4 features got " + str(x)
        check = False
    else:
        message = "Success"
        check = True
    
    return parameters, message, check 


class Predict(Resource):
    def get(self):
        result = {}
        result['success'] = False
        
        if flask.request.files.get("input_params"):
            args = parser.parse_args()
            input_params, message, check = input_check(args['input_params'])
            if check:
                predict = model.predict(input_params)
                result['prediction'] = list(predict)
                result['success'] = True
                result['message'] = message
            else:
                result['prediction'] = []
                result['message'] = message
            
    return flask.jsonify()

api.add_resource(Predict, '/')

if __name__ =='__main__':
    load_model()
    api.run(debug = True)
