# Importing the necessary Python modules.
import streamlit as st
from function import get_feat_data, load_data, load_values, chart_plot, prediction, model_selection
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

# Add title
st.title('Glass Type prediction Web app')
st.sidebar.title('Glass Type prediction Web app')

df = load_data()
X, y, X_train, X_test, y_train, y_test = load_values()

nav = st.sidebar.radio("Menu", ("Home", "Prediction", "Graph", "Contact Us"))

if nav == "Home":
	st.image("welcome.jpg")
	st.markdown("### Check the data used")
	raw_data = st.checkbox('Show raw data')
	if raw_data:
		st.dataframe(df, width=600)

elif nav == "Prediction":
	st.sidebar.markdown("## Give input data:")
	feat_list = get_feat_data(df)
	model, score = model_selection(X_train, y_train)
	button = st.button("Classify")
	if button:
		predicted = prediction(model, feat_list)
		st.success("Predicted sucessfully!")
		st.markdown("### Prediction:")
		st.success(predicted)
		st.set_option('deprecation.showPyplotGlobalUse', False)
		plot_confusion_matrix(model, X_test, y_test)
		st.pyplot()

elif nav == "Graph":
	st.sidebar.subheader('Visulisation Selector')
	t_chart = ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot')
	plot_list = st.sidebar.multiselect('Select the chart/plot', t_chart)
	chart_plot(plot_list, df)

else:
	st.balloons()
	st.header('Contact Us')
	st.markdown('''### Name:
	Shishir Shekhar''')
	st.markdown('''### Email:
	sspdav02@gmail.com''')
	st.markdown('''### GitHub: [ShishirShekhar](https://github.com/ShishirShekhar/)''')
