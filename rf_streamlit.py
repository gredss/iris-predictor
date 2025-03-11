import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
try:
    model = joblib.load('RF_class.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def main():
    st.title('Machine Learning Model Deployment')

    # Add user input components for 4 features
    sepal_length = st.slider('Sepal Length', 0.0, 10.0, 5.0)
    sepal_width = st.slider('Sepal Width', 0.0, 10.0, 3.0)
    petal_length = st.slider('Petal Length', 0.0, 10.0, 4.0)
    petal_width = st.slider('Petal Width', 0.0, 10.0, 1.0)

    if st.button('Make Prediction'):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
