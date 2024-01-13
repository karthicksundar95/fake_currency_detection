# -*- coding: utf-8 -*-
'''
    flask API calls that allows user to do single
    predict or batch prediction to detect fake currencies

@author Karthick Sundar C K
'''
from flask import Flask, request
from main import FakeCurrencyDetection
from flasgger import Swagger


fcd_app = Flask(__name__)
Swagger(fcd_app)

@fcd_app.route('/')
def welcome():
    """
    Default API call when the application starts with a welcome text
    :return: Welcome text message
    """
    return "WELCOME! to our application to detect fake currencies"

@fcd_app.route('/predict')
def predict():
    """Let's Authenticate the Banks Note
        This is using docstrings for specifications.
        ---
        parameters:
          - name: variance
            in: query
            type: number
            required: true
          - name: skewness
            in: query
            type: number
            required: true
          - name: kurtosis
            in: query
            type: number
            required: true
          - name: entropy
            in: query
            type: number
            required: true
        responses:
            200:
                description: The output values

        """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    kurtosis = request.args.get('kurtosis')
    entropy = request.args.get('entropy')
    out_of_sample_test = [[variance, skewness, kurtosis, entropy]]
    print(out_of_sample_test,"%%%%")
    fake_detector = FakeCurrencyDetection(mode='single_input_test',
                                          model_path="./fake_currency_detection_model.pkl")
    # out_of_sample_test = [[3, 2, 1, 1]]
    fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
    print("^^^",fake_detector.detection)
    return "The given currency is "+str(fake_detector.detection)


@fcd_app.route('/predict_file',methods=['POST'])
def predict_file():
    """Let's Authenticate the Banks Note
        This is using docstrings for specifications.
        ---
        parameters:
          - name: file
            in: formData
            type: file
            required: true

        responses:
            200:
                description: The output values

    """
    file_path = request.files.get('file')
    print("***", file_path)
    fake_detector1 = FakeCurrencyDetection(data_path=request.files.get('file'),
                                           mode='file_test',
                                           model_path="./fake_currency_detection_model.pkl")

    return str(list(fake_detector1.predicted_output))


if __name__ ==  "__main__":
    fcd_app.run(port='8000')
