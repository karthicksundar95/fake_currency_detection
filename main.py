####  main script which controls the flow of the application

# importing necessary packages
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class fake_currency_detection():

    def __init__(self,mode,data_path=None,model_path=None):
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
        self.currency_data = pd.read_csv(self.file_path)
        print(">> inside the read_input_csv_data method <<")


    def split_data_for_train_test(self):
        X = self.currency_data.iloc[:,:-1]
        y = self.currency_data.iloc[:,-1]
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        print(">> finished splitting the data {} into train {} and test {} <<".format(len(self.currency_data),
                                                                                      X.shape,y.shape))
    def build_model(self):
        self.classifier = RandomForestClassifier()
        self.classifier.fit(self.X_train, self.y_train)

    def predict_after_train(self):
        self.prediction = self.classifier.predict(self.X_test)

    def predict_from_loaded_model(self,test_data):
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
        self.score = accuracy_score(self.prediction, self.y_test)

    def export_model(self):
        joblib.dump(self.classifier, "./fake_currency_detection_model.pkl")

    def load_model(self, model_path):
        self.loaded_model = joblib.load(model_path)
        print("Model loaded successfully")

    def predict_out_of_sample(self,model_path, test_data):
        self.load_model(model_path)
        self.predict_from_loaded_model(test_data)


# uncomment the below section and run this file alone, inorder to train and validate the model
if __name__ ==  "__main__":
    ########### train val and single input test
    fake_detector = fake_currency_detection(data_path="{}/BankNote_Authentication.csv".format(Path.cwd()),
                                            mode='train')
    print("The accuracy of the model is {}".format(fake_detector.score * 100))
    out_of_sample_test = [[3,2,1,1]]
    fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
    print("The given curreny with features {} is {}".format(out_of_sample_test, fake_detector.detection))


    ########### file test
    fake_detector1 = fake_currency_detection(data_path="{}/BankNote_Authentication.csv".format(Path.cwd()),
                                            mode='file_test',model_path="./fake_currency_detection_model.pkl")

    print(fake_detector1.predicted_output)



    ########## single input test without train
    fake_detector = fake_currency_detection(mode='single_input_test',
                                            model_path="./fake_currency_detection_model.pkl")
    out_of_sample_test = [[3, 2, 1, 1]]
    fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
    print(fake_detector.predicted_output)

