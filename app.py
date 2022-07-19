"""Defines a simple, interactive Streamlit dashboard that analyzes
Motor vehicle collisions in New York City
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px


DATA_URL = ("Motor_Vehicle_Collisions_-_Crashes.csv")
st.title("Motor Vehicle Collisions in New York City")
st.markdown("This is a Streamlit dashboard that can be used \
            to analyze motor vehicle collisions in NYC")


@st.cache(persist=True)
def load_data(nrows):
    """Loads the data into a pandas DataFrame.
    Uses the st.cache decorator to optimize performance.
    The decorator computes the function and stores the return
    for faster retrieval of the data.

    Args:
        nrows: The number of rows to load
    """
    data = pd.read_csv(DATA_URL,
                       nrows=nrows,
                       parse_dates=[['CRASH_DATE', 'CRASH_TIME']])
    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)
    return data


data = load_data(100000)
original_data = data

# Find areas where most people are injured in NYC motor vehicle collisions
st.header("Where are the most people injured in NYC?")
st.markdown("Move the slider to pick the number of people to check\
            The map shows the spots where most people are injured")
injured_people = st.slider("Number of people injured in collisions", 0, 19)
# use pandas instead to get the maximum number of injured people
q = data.query("injured_persons >= @injured_people")[["latitude", "longitude"]]
midpoint = (np.average(q['latitude']), np.average(q['longitude']))
# st.map(q.dropna(how="any"), zoom=13)

st.write(pdk.Deck(
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
                q.dropna(how="any"),
                pickable=True,
                radius_scale=6,
                radius_min_pixels=6,
                radius_max_pixels=100,
                get_fill_color=[255, 140, 0],
                get_position=['longitude', 'latitude'],
            ),
        ],
))

# Check number of collisiona during a given time of the day

st.header("How many collisions occur during a given time of day?")
st.markdown("Move the slider to pick an hour")
hour = st.slider("Hour to look at", 0, 23)
data = data[data['date/time'].dt.hour == hour]

st.markdown("Vehicle collisions between %i:00 \
            and %i:00" % (hour, (hour + 1) % 24))
midpoint = (np.average(data['latitude']), np.average(data['longitude']))

st.write(pdk.Deck(
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
            data=data[['date/time', 'latitude', 'longitude']],
            get_position=['longitude', 'latitude'],
            radius=100,
            extruded=True,
            pickable=True,
            elevation_scale=4,
            elevation_range=[0, 1000],
        ),
    ],
))

st.subheader("Minute breakdown between %i:00 \
             and %i:00" % (hour, (hour + 1) % 24))
filtered = data[
    (data['date/time'].dt.hour >= hour) &
    (data['date/time'].dt.hour < (hour + 1))
]
hist = np.histogram(filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
fig = px.bar(chart_data,
             x='minute',
             y='crashes',
             hover_data=['minute', 'crashes'],
             height=400)
st.write(fig)

st.header("Top 5 dangerous streets by affected type")
st.markdown("Affected type of people include Pedestrians, \
            Cyclists and Motorists.")
select = st.selectbox('Affected type of people',
                      ['Pedestrians', 'Cyclists', 'Motorists'])

if select == 'Pedestrians':
    st.write(original_data.query("injured_pedestrians >= 1")
             [["on_street_name", "injured_pedestrians"]]
             .sort_values(by=['injured_pedestrians'],
             ascending=False).dropna(how='any')[:5])
elif select == 'Cyclists':
    st.write(original_data.query("injured_cyclists >= 1")
             [["on_street_name", "injured_cyclists"]]
             .sort_values(by=['injured_cyclists'],
             ascending=False).dropna(how='any')[:5])
else:
    st.write(original_data.query("injured_motorists >= 1")
             [["on_street_name", "injured_motorists"]]
             .sort_values(by=['injured_motorists'],
             ascending=False).dropna(how='any')[:5])


if st.checkbox('Show Raw Data', False):
    st.subheader('Raw Data')
    st.write(data)
