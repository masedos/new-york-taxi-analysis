# New York Taxi Analysis

# Importing libraries
import streamlit as st
import pandas as pd
from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Background color
st.markdown(
      """
      <style>
      .main { background-color: #FFDCA3; }
      </style>
      """,
      unsafe_allow_html=True
  )


@st.cache
def get_data(filename):
	taxi_data = pd.read_csv(filename)
	return taxi_data


header = st.beta_container()
with header:
	st.title('New York Taxi Analysis')
	st.text('Predict the average money spent on taxi rides for each region of New York')


dataset = st.beta_container()
with dataset: 
	st.header('New York Taxi Dataset')
	st.markdown('The dataset original can be found in [NYC Taxi and Limousine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)')

	# Import dataset
	taxi_data = get_data('data/taxi_data.csv')

	st.header('Sample of the dataset')
	st.write(taxi_data.head())

	# Bar Chart
	st.subheader('Pick-up location ID distribution on the NYC Taxi Dataset')
	population_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
	st.bar_chart(population_dist)


features = st.beta_container()
with features:
	st.header('The description features of the dataset')
	st.write("""
		This data dictionary describes yellow taxi trip data. 
		For a dictionary describing green taxi data, or a map of the TLC Taxi Zones, 
		please visit http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml.
		""")

model_training = st.beta_container()
with model_training:
	st.header('Time to train the model!')
	st.text('Here you get to choose the hyperparameters of the model')

	select_col, disp_col = st.beta_columns(2)

	max_depth = select_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

	n_estimators = select_col.selectbox('How many trees should there be?', options=[100, 200, 300, 'No limit'], index=0)

	#options = taxi_data.columns
	population_dist = pd.DataFrame(taxi_data[['passenger_count','trip_distance','RatecodeID',
											'store_and_fwd_flag','PULocationID','DOLocationID',
											'payment_type','fare_amount','extra','mta_tax',
											'tip_amount','tolls_amount','improvement_surcharge',
											'total_amount','congestion_surcharge']])
	options = population_dist.columns
	

	with select_col:
	    level_radio = st.radio("Here is a list of features in my data", options)


	if n_estimators == 'No limit':
		regr = RandomForestRegressor(max_depth=max_depth)
	else:
		regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


	X = taxi_data[[level_radio]]
	y = taxi_data[['trip_distance']]

	regr.fit(X, y)
	prediction = regr.predict(y)

	disp_col.subheader('Mean absolute error of the model is:')
	disp_col.write(mean_absolute_error(y, prediction))

	disp_col.subheader('Mean squared error of the model is:')
	disp_col.write(mean_squared_error(y, prediction))

	disp_col.subheader('Root mean squared error is:')
	disp_col.write(sqrt(mean_squared_error(y, prediction)))

	disp_col.subheader('R squared score of the model is:')
	disp_col.write(r2_score(y, prediction))