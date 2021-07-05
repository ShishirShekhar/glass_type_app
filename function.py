"""
This module contains all the functions required for the web application
"""
# Improt necceary module.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
"""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score """

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
	glass_df = load_data()
	# Creating the features data-frame holding all the columns except the last column.
	X = glass_df.iloc[:, :-1]
	# Creating the target series that holds last column.
	y = glass_df['GlassType']
	# Spliting the data into training and testing sets.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
	return X, y, X_train, y_train

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

def chart_plot(plot_list, glass_df):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	if 'Correlation Heatmap' in plot_list:
		# plot correlation heatmap
		st.subheader('Correlation Heatmap')
		fig = plt.figure(figsize=(12, 5))
		sns.heatmap(glass_df.corr(), annot=True)
		st.pyplot()
	if 'Line Chart' in plot_list:
		# plot line chart
		st.subheader('Line Chart')
		st.line_chart(glass_df)
	if 'Area Chart' in plot_list:
		# plot area chart
		st.subheader('Area Chart')
		st.area_chart(glass_df) 
	if 'Count Plot' in plot_list:
		# plot count plot
		st.subheader('Count Plot')
		sns.countplot(glass_df.iloc[:, -1])
		st.pyplot()
	if 'Pie Chart' in plot_list:
		# plot pie chart
		st.subheader('Pie Chart')
		pie_data = glass_df['GlassType'].value_counts()
		plt.pie(pie_data, labels=pie_data.index, autopct='%1.2f%%', startangle=30)
		st.pyplot()
	if 'Box Plot' in plot_list:
		# plot box plot
		st.subheader('Box Plot')
		material = st.selectbox('Select variable for boxplot', glass_df.columns)
		sns.boxplot(glass_df[material])
		st.pyplot()
