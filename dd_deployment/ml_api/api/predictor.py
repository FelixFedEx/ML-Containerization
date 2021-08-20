# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
from __future__ import absolute_import
from __future__ import print_function

import flask
import pandas as pd

from duplicate_detection_model.predict import make_prediction


def predict(input_object):
    df = pd.DataFrame({'short_desc': [input_object.get('ShortDesc')],
                       'create_date': [input_object.get('CreateDate')],
                       'sub_sys': [input_object.get('SUBSYS')]})

    subject = make_prediction(input_data=df)
    version, package_name, error_message = subject['version'], subject['package_name'], subject['errors']
    dup_obs = None
    dup_score = None
    
    if not error_message:
        # 1. for predicted subsys to get first result
        dup_obs = [idx[0] for idx in subject['predictions'][0]]
        dup_score = [idx[1] for idx in subject['predictions'][0]]

        # 2. for key words and their predicted attention weights

    return dup_obs, dup_score, version, package_name, error_message


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load all of the models successfully."""
    status = 200  # if all(OwnerPredictor.load_models()) else 404

    return flask.Response(response='Ping Succeeds!\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    # Convert from JSON to pandas
    if flask.request.content_type == 'application/json':
        dict_api_text = flask.request.get_json()
    else:
        return flask.Response(response=f'This predictor only supports JSON data: ', status=415, mimetype='text/plain')

    # Do the prediction
    dup_obs, dup_score, version, package_name, error_message = predict(dict_api_text)

    # output
    if not error_message:
        pred_result = {
            'statusCode': 200,
            "dupOBS": dup_obs,
            "dupScore": dup_score,
            "modelName": package_name,
            "modelVersion": version
        }

        resp = flask.jsonify(pred_result)
        resp.status_code = 200
        return resp
    resp = flask.jsonify({'statusCode': 500,
                          "errorMsg": str(error_message[0]),
                          "modelName": package_name,
                          "modelVersion": version})
    resp.status_code = 200
    return resp
