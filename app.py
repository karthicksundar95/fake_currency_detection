# -*- coding: utf-8 -*-
'''
    flask API calls that allows user to do single
    predict or batch prediction to detect fake currencies

@author Karthick Sundar C K
'''
from flask import Flask, request
from main import FakeCurrencyDetection


fcd_app = Flask(__name__)

@fcd_app.route('/')
def welcome():
    """
    Default API call when the application starts with a welcome text
    :return: Welcome text message
    """
    return "WELCOME! to our application to detect fake currencies"

@fcd_app.route('/predict')
def predict():
    """
    API to send single input for prediction
    :return: Yes or No if the currency is fake
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    kurtosis = request.args.get('kurtosis')
    entropy = request.args.get('entropy')
    out_of_sample_test = [[variance, skewness, kurtosis, entropy]]
    fake_detector = FakeCurrencyDetection(mode='single_input_test',
                                          model_path="./fake_currency_detection_model.pkl")
    fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
    return fake_detector.detection


@fcd_app.route('/predict_file',methods=['POST'])
def predict_file():
    """
    API call to post a file with inputs and generated output for all
    :return: array of values saying 0 or 1 for not-fake and fake detection
    """
    file_path = request.files.get('file')
    print("***", file_path)
    fake_detector1 = FakeCurrencyDetection(data_path=request.files.get('file'),
                                           mode='file_test',
                                           model_path="./fake_currency_detection_model.pkl")

    return str(list(fake_detector1.predicted_output))


if __name__ ==  "__main__":
    fcd_app.run()
    # fake_detector = fake_currency_detection("{}/BankNote_Authentication.csv".format(Path.cwd()))
    # print("The accuracy of the model is {}".format(fake_detector.score * 100))
    # out_of_sample_test = [[3,2,1,1]]
    # fake_detector.predict_out_of_sample
    #               ("./fake_currency_detection_model.pkl", out_of_sample_test)
    # print("The given curreny with features {} is {}".format
    #       (out_of_sample_test, fake_detector.detection))
