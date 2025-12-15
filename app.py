"""
NYC Motor Vehicle Collisions Dashboard
A Streamlit dashboard that analyzes motor vehicle collisions in New York City
"""

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
import tarfile

@st.cache_data
def uncompress():
    with tarfile.open("Motor_Vehicle_Collisions_-_Crashes.csv.tar.gz") as f:
        f.extractall(".")

uncompress()
DATA_URL = ("Motor_Vehicle_Collisions_-_Crashes.csv")

st.title("Motor Vehicle Collisions in New York City")
st.markdown("This is a Streamlit dashboard that can be used to analyze motor vehicle collisions in NYC")

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL,
                       nrows=nrows,
                       parse_dates=[['CRASH_DATE', 'CRASH_TIME']])
    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    def to_lower(x):
        return str(x).lower()
    lowercase = to_lower
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)
    return data

data = load_data(100000)
original_data = data.copy()

# Date Range Filter
st.sidebar.header("Filters")
min_date = data['date/time'].min().date()
max_date = data['date/time'].max().date()
date_start, date_end = st.sidebar.date_input("Date Range", [min_date, max_date])

data = data[(data['date/time'] >= pd.to_datetime(date_start)) &
            (data['date/time'] <= pd.to_datetime(date_end))]

# KPIs
if not data.empty:
    col1, col2, col3, col4 = st.columns(4)
    total_collisions = len(data)
    avg_injuries = data['number_of_persons_injured'].mean()
    peak_hour = data['date/time'].dt.hour.mode()[0]
    injury_rate = (data['number_of_persons_injured'] > 0).mean() * 100

    with col1:
        st.metric("Total Collisions", f"{total_collisions:,}")
    with col2:
        st.metric("Avg Injuries per Crash", f"{avg_injuries:.1f}")
    with col3:
        st.metric("Peak Hour", f"{peak_hour}:00")
    with col4:
        st.metric("Injury Rate", f"{injury_rate:.1f}%")

# Find areas where most people are injured in NYC motor vehicle collisions
st.header("Where are the most people injured in NYC?")
st.markdown("Move the slider to pick the number of people to check. The map shows the spots where most people are injured")
injured_people = st.slider("Number of people injured in collisions", 0, 19)
q = data.query("number_of_persons_injured >= @injured_people")[["latitude", "longitude"]]
midpoint = (np.average(q['latitude']), np.average(q['longitude']))

st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            'latitude': midpoint[0],
            'longitude': midpoint[1],
            'zoom': 11,
            'pitch': 50,
        },
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                q,
                pickable=True,
                radius_scale=6,
                radius_min_pixels=6,
                radius_max_pixels=100,
                get_fill_color=[255, 140, 0],
                get_position=['longitude', 'latitude'],
            ),
        ],
))

# Check number of collisions during a given time of the day
st.header("How many collisions occur during a given time of day?")
st.markdown("Move the slider to pick an hour")
hour = st.slider("Hour to look at", 0, 23)
filtered_hourly = data[data['date/time'].dt.hour == hour]  # Use copy to avoid overwriting

st.markdown("Vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))
midpoint = (np.average(filtered_hourly['latitude']), np.average(filtered_hourly['longitude']))

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            filtered_hourly[['date/time', 'latitude', 'longitude']],
            get_position=['longitude', 'latitude'],
            radius=100,
            extruded=True,
            pickable=True,
            elevation_scale=4,
            elevation_range=[0, 1000],
        ),
    ],
))

st.subheader("Minute breakdown between %i:00 and %i:00" % (hour, (hour + 1) % 24))
filtered = filtered_hourly[
    (filtered_hourly['date/time'].dt.hour >= hour) &
    (filtered_hourly['date/time'].dt.hour < (hour + 1))
]
hist = np.histogram(filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
fig = px.bar(chart_data,
             x='minute',
             y='crashes',
             hover_data=['minute', 'crashes'],
             height=400)
st.plotly_chart(fig)

st.header("Top 5 dangerous streets by affected type")
st.markdown("Affected type of people include Pedestrians, Cyclists and Motorists.")
select = st.selectbox('Affected type of people',
                      ['Pedestrians', 'Cyclists', 'Motorists'])

if select == 'Pedestrians':
    st.dataframe(original_data.query("number_of_pedestrians_injured >= 1")
             [["on_street_name", "number_of_pedestrians_injured"]]
             .sort_values(by=['number_of_pedestrians_injured'],
             ascending=False).dropna(how='any')[:5])
elif select == 'Cyclists':
    st.dataframe(original_data.query("number_of_cyclist_injured >= 1")
             [["on_street_name", "number_of_cyclist_injured"]]
             .sort_values(by=['number_of_cyclist_injured'],
             ascending=False).dropna(how='any')[:5])
else:
    st.dataframe(original_data.query("number_of_motorist_injured >= 1")
             [["on_street_name", "number_of_motorist_injured"]]
             .sort_values(by=['number_of_motorist_injured'],
             ascending=False).dropna(how='any')[:5])


if st.checkbox('Show Raw Data', False):
    st.subheader('Raw Data')
    st.dataframe(data)
