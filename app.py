"""Defines a simple, interactive Streamlit dashboard that analyzes
Motor vehicle collisions in New York City
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import tarfile
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="NYC Motor Vehicle Collisions Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(persist=True, show_spinner=False)
def uncompress():
    """Uncompress the data file if needed"""
    try:
        with tarfile.open("Motor_Vehicle_Collisions_-_Crashes.csv.tar.gz") as f:
            f.extractall(".")
    except FileNotFoundError:
        st.warning("Compressed file not found. Assuming CSV already exists.")


@st.cache_data(persist=True, show_spinner="Loading collision data...")
def load_data(nrows):
    """Loads the data into a pandas DataFrame with optimized caching.

    Args:
        nrows: The number of rows to load

    Returns:
        pd.DataFrame: Processed collision data
    """
    DATA_URL = "Motor_Vehicle_Collisions_-_Crashes.csv"

    data = pd.read_csv(
        DATA_URL,
        nrows=nrows,
        parse_dates=[['CRASH_DATE', 'CRASH_TIME']]
    )

    # Drop rows without location data
    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    # Normalize column names
    data.rename(str.lower, axis='columns', inplace=True)
    data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)

    # Add additional time-based columns for analysis
    data['hour'] = data['date/time'].dt.hour
    data['day_of_week'] = data['date/time'].dt.day_name()
    data['month'] = data['date/time'].dt.month_name()
    data['year'] = data['date/time'].dt.year

    return data


@st.cache_data
def calculate_kpis(data):
    """Calculate key performance indicators from the collision data"""
    total_collisions = len(data)
    total_injured = data['injured_persons'].sum()
    total_killed = data['killed_persons'].sum()
    avg_injuries_per_collision = data['injured_persons'].mean()

    # Most dangerous hour
    collisions_by_hour = data.groupby('hour').size()
    most_dangerous_hour = collisions_by_hour.idxmax()

    # Most dangerous day
    collisions_by_day = data.groupby('day_of_week').size()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    collisions_by_day = collisions_by_day.reindex(day_order)
    most_dangerous_day = collisions_by_day.idxmax()

    return {
        'total_collisions': total_collisions,
        'total_injured': int(total_injured),
        'total_killed': int(total_killed),
        'avg_injuries': round(avg_injuries_per_collision, 2),
        'dangerous_hour': int(most_dangerous_hour),
        'dangerous_day': most_dangerous_day
    }


def display_kpis(kpis):
    """Display KPI metrics in a clean layout"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Collisions",
            value=f"{kpis['total_collisions']:,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Total Injuries",
            value=f"{kpis['total_injured']:,}",
            delta=None
        )

    with col3:
        st.metric(
            label="Total Fatalities",
            value=f"{kpis['total_killed']:,}",
            delta=None
        )

    with col4:
        st.metric(
            label="Avg Injuries/Collision",
            value=f"{kpis['avg_injuries']}",
            delta=None
        )

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        st.info(f"🕐 **Most Dangerous Hour:** {kpis['dangerous_hour']}:00 - {kpis['dangerous_hour']+1}:00")
    with col6:
        st.info(f"📅 **Most Dangerous Day:** {kpis['dangerous_day']}")


def create_time_series_chart(data):
    """Create a time series chart of collisions over time"""
    daily_collisions = data.groupby(data['date/time'].dt.date).size().reset_index()
    daily_collisions.columns = ['date', 'collisions']

    fig = px.line(
        daily_collisions,
        x='date',
        y='collisions',
        title='Daily Collision Trends',
        labels={'collisions': 'Number of Collisions', 'date': 'Date'}
    )
    fig.update_layout(hovermode='x unified')
    return fig


def create_hourly_distribution(data):
    """Create hourly distribution chart"""
    hourly_data = data.groupby('hour').size().reset_index()
    hourly_data.columns = ['hour', 'collisions']

    fig = px.bar(
        hourly_data,
        x='hour',
        y='collisions',
        title='Collisions by Hour of Day',
        labels={'collisions': 'Number of Collisions', 'hour': 'Hour'},
        color='collisions',
        color_continuous_scale='Reds'
    )
    return fig


def create_day_distribution(data):
    """Create day of week distribution chart"""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_data = data.groupby('day_of_week').size().reindex(day_order).reset_index()
    daily_data.columns = ['day', 'collisions']

    fig = px.bar(
        daily_data,
        x='day',
        y='collisions',
        title='Collisions by Day of Week',
        labels={'collisions': 'Number of Collisions', 'day': 'Day of Week'},
        color='collisions',
        color_continuous_scale='Blues'
    )
    return fig


# Main app
def main():
    # Uncompress data if needed
    uncompress()

    # Header
    st.title("🚗 NYC Motor Vehicle Collisions Dashboard")
    st.markdown("""
    This interactive dashboard analyzes motor vehicle collisions in New York City.
    Explore patterns, identify dangerous locations, and understand collision trends.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    nrows = st.sidebar.number_input(
        "Number of rows to load",
        min_value=1000,
        max_value=500000,
        value=100000,
        step=10000,
        help="Loading more rows provides more data but may be slower"
    )

    # Load data
    with st.spinner("Loading collision data..."):
        data = load_data(nrows)
        original_data = data.copy()

    # Calculate and display KPIs
    st.header("📊 Key Performance Indicators")
    kpis = calculate_kpis(data)
    display_kpis(kpis)

    # Time-based analysis
    st.header("📈 Collision Trends Over Time")
    tab1, tab2, tab3 = st.tabs(["Daily Trends", "Hourly Distribution", "Weekly Distribution"])

    with tab1:
        st.plotly_chart(create_time_series_chart(data), use_container_width=True)

    with tab2:
        st.plotly_chart(create_hourly_distribution(data), use_container_width=True)

    with tab3:
        st.plotly_chart(create_day_distribution(data), use_container_width=True)

    st.markdown("---")

    # Geographic analysis - Injuries
    st.header("🗺️ Where are the most people injured in NYC?")
    st.markdown("Use the slider to filter locations by minimum number of injuries.")

    col1, col2 = st.columns([2, 1])
    with col1:
        injured_people = st.slider(
            "Minimum number of people injured",
            min_value=0,
            max_value=19,
            value=1
        )

    with col2:
        st.metric("Locations Found", len(data.query("injured_persons >= @injured_people")))

    q = data.query("injured_persons >= @injured_people")[["latitude", "longitude"]]

    if len(q) > 0:
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
    else:
        st.warning("No collisions found with the selected criteria.")

    st.markdown("---")

    # Time-based geographic analysis
    st.header("🕐 Collisions by Time of Day")
    st.markdown("Explore collision patterns throughout the day.")

    col1, col2 = st.columns([2, 1])
    with col1:
        hour = st.slider("Hour to analyze", min_value=0, max_value=23, value=12)

    hour_data = original_data[original_data['hour'] == hour]

    with col2:
        st.metric("Collisions in this hour", len(hour_data))

    st.markdown(f"**Vehicle collisions between {hour}:00 and {(hour + 1) % 24}:00**")

    if len(hour_data) > 0:
        midpoint = (np.average(hour_data['latitude']), np.average(hour_data['longitude']))

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
                    data=hour_data[['date/time', 'latitude', 'longitude']],
                    get_position=['longitude', 'latitude'],
                    radius=100,
                    extruded=True,
                    pickable=True,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                ),
            ],
        ))

        # Minute breakdown
        st.subheader(f"Minute-by-minute breakdown: {hour}:00 - {(hour + 1) % 24}:00")
        filtered = original_data[
            (original_data['hour'] >= hour) &
            (original_data['hour'] < (hour + 1))
        ]
        hist = np.histogram(filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]
        chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
        fig = px.bar(
            chart_data,
            x='minute',
            y='crashes',
            hover_data=['minute', 'crashes'],
            height=400,
            labels={'crashes': 'Number of Crashes', 'minute': 'Minute'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No collisions found for this hour.")

    st.markdown("---")

    # Dangerous streets analysis
    st.header("⚠️ Top 5 Most Dangerous Streets")
    st.markdown("Identify the most dangerous streets by affected population type.")

    select = st.selectbox(
        'Select affected population type',
        ['Pedestrians', 'Cyclists', 'Motorists']
    )

    if select == 'Pedestrians':
        top_streets = (original_data.query("injured_pedestrians >= 1")
                      [["on_street_name", "injured_pedestrians"]]
                      .sort_values(by=['injured_pedestrians'], ascending=False)
                      .dropna(how='any')[:5])
        st.dataframe(top_streets, use_container_width=True)

    elif select == 'Cyclists':
        top_streets = (original_data.query("injured_cyclists >= 1")
                      [["on_street_name", "injured_cyclists"]]
                      .sort_values(by=['injured_cyclists'], ascending=False)
                      .dropna(how='any')[:5])
        st.dataframe(top_streets, use_container_width=True)

    else:
        top_streets = (original_data.query("injured_motorists >= 1")
                      [["on_street_name", "injured_motorists"]]
                      .sort_values(by=['injured_motorists'], ascending=False)
                      .dropna(how='any')[:5])
        st.dataframe(top_streets, use_container_width=True)

    st.markdown("---")

    # Contributing factors analysis
    st.header("🔍 Contributing Factors Analysis")
    if 'contributing_factor_vehicle_1' in original_data.columns:
        factors = (original_data['contributing_factor_vehicle_1']
                  .value_counts()
                  .head(10))

        fig = px.bar(
            x=factors.values,
            y=factors.index,
            orientation='h',
            title='Top 10 Contributing Factors',
            labels={'x': 'Number of Collisions', 'y': 'Contributing Factor'},
            color=factors.values,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Raw data viewer
    st.markdown("---")
    with st.expander("🔍 View Raw Data"):
        st.dataframe(original_data, use_container_width=True)

        # Download button
        csv = original_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'nyc_collisions_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Data source: NYC Open Data - Motor Vehicle Collisions</p>
        <p>Dashboard built with Streamlit | Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
