"""
This module contains all the functions required for the web application
"""
# Improt necceary module.
from altair.vegalite.v4.schema.channels import Y
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

@st.cache()
def load_values():
	# Load data
	df = load_data()
	# Creating the features data-frame holding all the columns except the last column.
	X = df.iloc[:, :-1]
	# Creating the target series that holds last column.
	y = df['GlassType']
	# Spliting the data into training and testing sets.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
	return X, y, X_train, X_test, y_train, y_test

@st.cache()
def prediction(model, target_list):
	y_pred = model.predict(target_list)
	glass_type = y_pred[0]
	if glass_type == 1:
		return 'building windows float processed'
	elif glass_type == 2:
		return 'building windows non float processed'
	elif glass_type == 3:
		return 'vehicle windows float processed'
	elif glass_type == 4:
		return 'vehicle windows non float processed'
	elif glass_type == 5:
		return 'containers'
	elif glass_type == 6:
		return 'tableware'
	else:
		return 'headlamp'

def chart_plot(plot_list, df):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	if 'Correlation Heatmap' in plot_list:
		# plot correlation heatmap
		st.subheader('Correlation Heatmap')
		fig = plt.figure(figsize=(12, 5))
		sns.heatmap(df.corr(), annot=True)
		st.pyplot()
	if 'Line Chart' in plot_list:
		# plot line chart
		st.subheader('Line Chart')
		st.line_chart(df)
	if 'Area Chart' in plot_list:
		# plot area chart
		st.subheader('Area Chart')
		st.area_chart(df) 
	if 'Count Plot' in plot_list:
		# plot count plot
		st.subheader('Count Plot')
		sns.countplot(df.iloc[:, -1])
		st.pyplot()
	if 'Pie Chart' in plot_list:
		# plot pie chart
		st.subheader('Pie Chart')
		pie_data = df['GlassType'].value_counts()
		plt.pie(pie_data, labels=pie_data.index, autopct='%1.2f%%', startangle=30)
		st.pyplot()
	if 'Box Plot' in plot_list:
		# plot box plot
		st.subheader('Box Plot')
		material = st.selectbox('Select variable for boxplot', df.columns)
		sns.boxplot(df[material])
		st.pyplot()

@st.cache()
def prediction(model, feature_list):
    glass_type = model.predict([feature_list])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

def get_feat_data(df):
    RI = st.sidebar.slider("RI", float(df["RI"].min()), float(df["RI"].max()))
    Na = st.sidebar.slider("Na", float(df["Na"].min()), float(df["Na"].max()))
    Mg = st.sidebar.slider("Mg", float(df["Mg"].min()), float(df["Mg"].max()))
    Al = st.sidebar.slider("Al", float(df["Al"].min()), float(df["Al"].max()))
    Si = st.sidebar.slider("Si", float(df["Si"].min()), float(df["Si"].max()))
    K = st.sidebar.slider("K", float(df["K"].min()), float(df["K"].max()))
    Ca = st.sidebar.slider("Ca", float(df["Ca"].min()), float(df["Ca"].max()))
    Ba = st.sidebar.slider("Ba", float(df["Ba"].min()), float(df["Ba"].max()))
    Fe = st.sidebar.slider("Fe", float(df["Fe"].min()), float(df["Fe"].max()))
    feat_list = [RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]
    return feat_list

def model_selection(X_train, y_train):
	st.subheader("Select the Classifier")
	clf_list = ("Support Vector Machine", "Logistic Regression", "Random Forest Classifier")
	smodel = st.selectbox("Classifier", clf_list)
	st.subheader("Model Hyperparameter")
	if smodel == "Support Vector Machine":
		C = st.number_input("C", 0.01, 100.0)
		kernel = st.radio("Kernel", ("linear", "rbf", "poly"))
		if kernel == "rbf":
			gamma = st.number_input("Gamma", 0.01, 1.0)
			model, score = svc_model(X_train, y_train, kernel=kernel, C=C, gamma=gamma)
		elif kernel == "poly":
			gamma = st.number_input("Gamma", 0.01, 1.0)
			degree = st.number_input("Degree", 1, 6)
			model, score = svc_model(X_train, y_train, C, kernel, degree, gamma)
		else:
			model, score = svc_model(X_train, y_train, C, kernel="linear")
	elif smodel == "Random Forest Classifier":
		n_estimators = st.number_input("Number of tress", 1, 100)
		max_depth = st.number_input("Max Depth of tress", 1, 15)
		model, score = rfc_model(X_train, y_train, n_estimators, max_depth)
	else:
		C = st.number_input("C", 0.01, 100.0)
		model, score = lr_model(X_train, y_train, C)
	
	return model, score
			
@st.cache
def svc_model(X_train, y_train, C=1, kernel="linear", degree=3, gamma="scale"):
	svc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
	svc.fit(X_train, y_train)
	score = svc.score(X_train, y_train)
	return svc, score

@st.cache
def rfc_model(X_train, y_train, n_estimators, max_depth):
	rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
	rf_clf.fit(X_train, y_train)
	score = rf_clf.score(X_train, y_train)
	return rf_clf, score

@st.cache
def lr_model(X_train, y_train, C):
	lr = LogisticRegression(C=C)
	lr.fit(X_train, y_train)
	score = lr.score(X_train, y_train)
	return lr, score