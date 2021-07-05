# Importing the necessary Python modules.
import streamlit as st
from function import load_data, chart_plot


# Add title
st.title('Glass Type prediction Web app')
st.sidebar.title('Glass Type prediction Web app')

# Get data and show
glass_df = load_data()
raw_data = st.sidebar.checkbox('Show raw data')
if raw_data:
	st.dataframe(glass_df, width=600)

st.sidebar.subheader('Visulisation Selector')
plot_list = st.sidebar.multiselect('Select the chart/plot', ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))

chart_plot(plot_list, glass_df)
