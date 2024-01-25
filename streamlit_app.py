# -*- coding: utf-8 -*-
'''
    flask API calls that allows user to do single
    predict or batch prediction to detect fake currencies

@author Karthick Sundar C K
'''
import streamlit as st
from main import FakeCurrencyDetection

def welcome():
    """
    Default API call when the application starts with a welcome text
    :return: Welcome text message
    """
    return "WELCOME! to our application to detect fake currencies"

def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance","Type Here")
    skewness = st.text_input("skewness","Type Here")
    kurtosis = st.text_input("kurtosis","Type Here")
    entropy = st.text_input("entropy","Type Here")
    results = "The given currency is "
    if st.button("Predict"):
        out_of_sample_test = [[variance, skewness, kurtosis, entropy]]
        print(out_of_sample_test, "%%%%")
        fake_detector = FakeCurrencyDetection(mode='single_input_test',
                                              model_path="./fake_currency_detection_model.pkl")
        # out_of_sample_test = [[3, 2, 1, 1]]
        fake_detector.predict_out_of_sample("./fake_currency_detection_model.pkl", out_of_sample_test)
        print("^^^", fake_detector.detection)
        results = results + str(fake_detector.detection)
    st.success(results)
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()

