# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# App title
st.title("Iris Flower Classifier")

# Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["species"])

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Function to take user input from sidebar
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
    petal_length = st.sidebar.slider('Petal length', float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
    petal_width = st.sidebar.slider('Petal width', float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Show input data
st.subheader('User Input Parameters')
st.write(input_df)

# Train a RandomForest classifier on the iris dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())

# Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Show prediction
st.subheader('Prediction')
st.write(iris.target_names[prediction])

# Show prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)

# Accuracy of the model
st.subheader('Model Accuracy')
st.write(accuracy_score(y_test, clf.predict(X_test)))

# Plot feature importance
st.subheader('Feature Importance')
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot graph
fig, ax = plt.subplots()
ax.barh(iris.feature_names, importances[indices])
st.pyplot(fig)

# To run this app:
# 1. Save this file as `app.py`
# 2. Run `streamlit run app.py` in the terminal
