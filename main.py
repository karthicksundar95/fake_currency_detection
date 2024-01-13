# -*- coding: utf-8 -*-
'''
    Core fake currency class with methods to train and predict

@author Karthick Sundar C K
'''
# importing necessary packages
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class FakeCurrencyDetection():
    """
    Class to detect fake currency based on the feature values provided
    """

    def __init__(self,mode,data_path=None,model_path=None):
        """
        Contructor to define flow control for train
        single input test and file input test
        :param mode: train, single input test or file level
        :param data_path: location of the data to train or file prediction
        :param model_path: location of the trained model for prediction usage
        """
        if mode == "train":
            print("INSIDE TRAIN")
            self.file_path = data_path
            self.read_input_csv_data()
            self.split_data_for_train_test()
            self.build_model()
            self.predict_after_train()
            self.model_performance()
            self.export_model()
        elif mode == 'file_test':
            print("INSIDE TEST")
            self.file_path = data_path
            self.read_input_csv_data()
            self.load_model(model_path)
            self.predict_from_loaded_model(self.currency_data)
        elif mode == 'single_input_test':
            self.load_model(model_path)

    def read_input_csv_data(self):
        """
        Method to read the data from CSV file
        :return: pandas dataframe of the read data
        """
        self.currency_data = pd.read_csv(self.file_path)
        print(">> inside the read_input_csv_data method <<")

    def split_data_for_train_test(self):
        """
        Method to split the data into train and test
        :return: Train and test splits of independent and target columns
        """
        X = self.currency_data.iloc[:,:-1]
        y = self.currency_data.iloc[:,-1]
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split\
                                                            (X,y,test_size=0.2,random_state=42)
        print(">> finished splitting the data {} into train {} and test {} <<".
               format(len(self.currency_data),X.shape,y.shape))
    def build_model(self):
        """
        Method to build random forest classifier
        :return: model object
        """
        self.classifier = RandomForestClassifier()
        self.classifier.fit(self.X_train, self.y_train)

    def predict_after_train(self):
        """
        Method to validate the training the process on the test split
        :return: numpy array with predictions on test split
        """
        self.prediction = self.classifier.predict(self.X_test)

    def predict_from_loaded_model(self,test_data):
        """
        Method to do testing on unseen data using alread built model
        :param test_data: unseen data read from user
        :return: array with 0 or 1 indicating not-fake and fake
        """
        if len(test_data) == 1:
            self.predicted_output = self.loaded_model.predict(test_data)
            if self.predicted_output > 0.5:
                self.detection = "Fake"
            else:
                self.detection = "Not Fake"
        else:
            test_data = test_data.iloc[:, :-1]
            self.predicted_output = self.loaded_model.predict(test_data)
            print("PREDICTED OUTPUT:")
            print(self.predicted_output )

    def model_performance(self):
        """
        Method to calculate the accuracy of the model
        :return: accuracy measure
        """
        self.score = accuracy_score(self.prediction, self.y_test)

    def export_model(self):
        """
        Method to save the built model to a file location
        :return: model file
        """
        joblib.dump(self.classifier, "./fake_currency_detection_model.pkl")

    def load_model(self, model_path):
        """
        Method to load the model file from disc to RAM
        :param model_path: location of model to load
        :return: model object loaded
        """
        self.loaded_model = joblib.load(model_path)
        print("Model loaded successfully")

    def predict_out_of_sample(self,model_path, test_data):
        """
        Method to control loading of model and prediction
        :param model_path: location of the model to be loaded
        :param test_data: unseen data to be predicted
        """
        self.load_model(model_path)
        self.predict_from_loaded_model(test_data)


# uncomment the below section and run this file alone, inorder to train and validate the model
if __name__ ==  "__main__":
    ########### train val and single input test
    fake_detector = FakeCurrencyDetection(data_path="{}/BankNote_Authentication.csv"
                                          .format(Path.cwd()), mode='train')
    print("The accuracy of the model is {}".format(fake_detector.score * 100))
    out_of_sample_test = [[3,2,1,1]]
    fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
    print("The given curreny with features {} is {}"
          .format(out_of_sample_test, fake_detector.detection))


    ########### file test
    fake_detector1 = FakeCurrencyDetection(data_path="{}/BankNote_Authentication.csv"
                                           .format(Path.cwd()),
                                           mode='file_test',
                                           model_path="./fake_currency_detection_model.pkl")

    print(fake_detector1.predicted_output)



    ########## single input test without train
    fake_detector = FakeCurrencyDetection(mode='single_input_test',
                                          model_path="./fake_currency_detection_model.pkl")
    out_of_sample_test = [[3, 2, 1, 1]]
    fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
    print(fake_detector.predicted_output)
