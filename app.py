import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import random
import keras
import tensorflow as tf

from transformers import AutoTokenizer
from transformers import TFDistilBertModel, AutoConfig

import streamlit as st
from twitter import twitter_model


def main():
    st.header('Twitter disater detector')
    directory = os.getcwd()
    weights_path= directory+"/custom_model.keras"
    model_test= twitter_model(weights_path)
    input_text=st.text_input("Please enter your sentence:", "type a word")
    prediction= np.round(model_test.predict(input_text))
    disaster= False
    if prediction==1:
        disaster= True
    if disaster:
        st.write("the text: '",input_text, "' means there is a DISASTER" )
    else:
        st.write("the text: '",input_text, "' means there is NO DISASTER" )
    
    

if __name__ == '__main__':
    main()