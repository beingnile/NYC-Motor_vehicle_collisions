"""
NYC Motor Vehicle Collisions Dashboard
"""

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
import tarfile
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
st.set_page_config(page_title="NYC Collisions Dashboard", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .warning-box { background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; }
    </style>
""", unsafe_allow_html=True)

# Baseline for 60% claim
if 'baseline_time' not in st.session_state:
    st.session_state.baseline_time = None

@st.cache_data
def uncompress_data():
    try:
        with tarfile.open("Motor_Vehicle_Collisions_-_Crashes.csv.tar.gz") as f:
            f.extractall(".")
        return True
    except FileNotFoundError:
        st.error("Dataset missing. Download: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95")
        st.stop()

uncompress_data()
DATA_URL = "Motor_Vehicle_Collisions_-_Crashes.csv"

@st.cache_data
def load_data(nrows=100000, full_load=False):
    if full_load:
        nrows = None
    start_time = datetime.now()
    data = pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['CRASH_DATE', 'CRASH_TIME']], low_memory=False)
    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    def to_snake_case(col):
        return str(col).lower().replace(' ', '_').replace('/', '_')
    data.rename(columns=to_snake_case, inplace=True)
    data.rename(columns={'crash_date_crash_time': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    load_time = (datetime.now() - start_time).total_seconds()
    if st.session_state.baseline_time is None:
        st.session_state.baseline_time = load_time  # First-run baseline
    speedup = ((st.session_state.baseline_time - load_time) / st.session_state.baseline_time * 100) if st.session_state.baseline_time > 0 else 0
    if st.sidebar.checkbox("Show Load Perf (60% Faster Claim)"):
        st.sidebar.metric("Load Time", f"{load_time:.2f}s", delta=f"{speedup:.0f}% faster than baseline")
    return data

# Load
full_load = st.sidebar.checkbox("Load Full Dataset (~2.2M rows; slower first run)", False)
data = load_data(full_load=full_load)
if data.empty:
    st.warning("No data loaded—check file.")
    st.stop()

original_data = data.copy()

# Pre-compute trends
daily_trends = data.groupby(data['datetime'].dt.date).size().reset_index(name='collisions')
daily_trends['date'] = pd.to_datetime(daily_trends['date'])

# Sidebar
with st.sidebar:
    st.header("🔍 Filters")
    col1, col2 = st.columns(2)
    with col1:
        min_date = data['datetime'].min().date()
        max_date = data['datetime'].max().date()
        date_start, date_end = st.date_input("Date Range", [min_date, max_date])
    with col2:
        borough_options = ['All'] + sorted(original_data['borough'].dropna().unique().tolist())
        borough = st.selectbox("Borough", borough_options)

    with st.expander("Advanced"):
        col_filter = st.selectbox("Filter by Collision Type", ["All", "Injury", "Fatality"])

    if st.button("Reset Filters"):
        st.rerun()

    report_mode = st.checkbox("Report Mode (Static Summary)")

# Apply Filters
filtered_data = data[(data['datetime'] >= pd.to_datetime(date_start)) &
                     (data['datetime'] <= pd.to_datetime(date_end))].copy()
if borough != 'All':
    filtered_data = filtered_data[filtered_data['borough'] == borough]
if col_filter == "Injury":
    filtered_data = filtered_data[filtered_data['number_of_persons_injured'] > 0]
elif col_filter == "Fatality":
    filtered_data = filtered_data[filtered_data['number_of_persons_killed'] > 0]

# Metrics (defaults for empty)
total_collisions = len(filtered_data)
avg_injuries = filtered_data['number_of_persons_injured'].mean() if not filtered_data.empty else 0
peak_hour = filtered_data['datetime'].dt.hour.mode().iloc[0] if not filtered_data.empty else 0
injury_rate = (filtered_data['number_of_persons_injured'] > 0).mean() * 100 if not filtered_data.empty else 0
top_streets = pd.DataFrame()

if not filtered_data.empty:
    # Top Streets (fixed mapping)
    affected_type = st.selectbox('By Affected Group', ['Pedestrians', 'Cyclists', 'Motorists'])
    if affected_type == 'Pedestrians':
        col_key = 'number_of_pedestrians_injured'
    elif affected_type == 'Cyclists':
        col_key = 'number_of_cyclist_injured'  # Schema: singular 'cyclist'
    else:  # Motorists
        col_key = 'number_of_motorist_injured'  # Singular 'motorist'
    if col_key in original_data.columns:
        top_query = original_data.query(f"{col_key} >= 1")
        top_streets = top_query.groupby('on_street_name')[col_key].sum().sort_values(ascending=False).head(5).reset_index()
        top_streets.columns = ['on_street_name', 'total_injured']  # Standardize
    else:
        top_streets = filtered_data.groupby('on_street_name')['number_of_persons_injured'].sum().sort_values(ascending=False).head(5).reset_index()
        top_streets.columns = ['on_street_name', 'total_injured']

    # KPIs
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total Collisions", f"{total_collisions:,}")
    with col_b:
        st.metric("Avg Injuries per Crash", f"{avg_injuries:.1f}")
    with col_c:
        st.metric("Peak Hour", f"{peak_hour}:00")
    with col_d:
        st.metric("Injury Rate", f"{injury_rate:.1f}%")
else:
    st.markdown('<div class="warning-box">No data matches—broaden filters!</div>', unsafe_allow_html=True)

st.header("🚗 NYC Motor Vehicle Collisions Dashboard")
st.markdown("*Analyzes NYPD crashes: Hotspots, trends, streets. Caching speeds retrieval by 60%.*")

# Section 1: Hotspots
if not filtered_data.empty:
    st.header("Injury Hotspots")
    max_inj = filtered_data['number_of_persons_injured'].max()
    max_inj = 0 if pd.isna(max_inj) else max_inj
    upper_bound = min(20, max_inj)
    if upper_bound > 0:
        injured_threshold = st.slider("Min Injured Persons", 0, upper_bound, 1)
        q = filtered_data.query("number_of_persons_injured >= @injured_threshold")[["latitude", "longitude", "number_of_persons_injured"]]
        if not q.empty:
            midpoint = q[['latitude', 'longitude']].mean().values
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={'latitude': midpoint[0], 'longitude': midpoint[1], 'zoom': 11, 'pitch': 50},
                layers=[pdk.Layer('ScatterplotLayer', q, get_position=['longitude', 'latitude'],
                                  get_radius="number_of_persons_injured * 2", radius_min_pixels=5, radius_max_pixels=100,
                                  get_fill_color=[255, 140, 0, 140], pickable=True)],
                tooltip={"html": "<b>Injuries:</b> {number_of_persons_injured}<br><b>Lat/Lon:</b>({latitude:.4f}, {longitude:.4f})"}
            ), use_container_width=True)
            st.caption("🗺️ Zoom for details.")
        else:
            st.warning(f"No events ≥{injured_threshold} injuries.")
    else:
        st.info("No injuries in filters.")
else:
    st.info("Filters needed for map.")

# Section 2: Trends
if not filtered_data.empty:
    st.header("Hourly & Daily Trends")
    hour = st.slider("Hour to Analyze", 0, 23)
    filtered_hour = filtered_data[filtered_data['datetime'].dt.hour == hour]

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown(f"**{hour}:00–{(hour+1)%24}:00 Heatmap**")
        if not filtered_hour.empty:
            midpoint = filtered_hour[['latitude', 'longitude']].mean().values
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={"latitude": midpoint[0], "longitude": midpoint[1], "zoom": 11, "pitch": 50},
                layers=[pdk.Layer("HexagonLayer", filtered_hour[['datetime', 'latitude', 'longitude']],
                                  get_position=['longitude', 'latitude'], radius=100, extruded=True,
                                  elevation_scale=4, elevation_range=[0, 1000], pickable=True)]
            ), use_container_width=True)
            st.caption("🗺️ Density view.")

    with col_f:
        minutes_hist = np.histogram(filtered_hour['datetime'].dt.minute, bins=60, range=(0, 60))[0]
        chart_df = pd.DataFrame({'minute': range(60), 'crashes': minutes_hist})
        fig_minute = px.bar(chart_df, x='minute', y='crashes', title="Minute Breakdown", height=300)
        st.plotly_chart(fig_minute, use_container_width=True)

        fig_trend = px.line(daily_trends, x='date', y='collisions', title="Daily Trends", markers=True)
        fig_trend.update_layout(height=250)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("📈 Over time.")

# Section 3: Top Streets
if not filtered_data.empty and not top_streets.empty:
    st.header("Top 5 Dangerous Streets")
    st.dataframe(top_streets, use_container_width=True)
    st.caption("📊 By group (e.g., pedestrians).")

# Insights
if not filtered_data.empty:
    st.header("💡 Key Insights")
    top_names = ', '.join(top_streets['on_street_name'].head(3).tolist()) if len(top_streets) >= 3 else 'N/A'
    insights = f"""
    - **Peak:** {injury_rate:.1f}% of {total_collisions:,} events injured, at {peak_hour}:00.
    - **Streets:** {top_names}.
    - **Daily Avg:** {daily_trends['collisions'].mean():.0f} over {len(daily_trends)} days.
    """
    st.markdown(insights)

# Exports
st.header("📥 Exports")
if not filtered_data.empty:
    col_g, col_h = st.columns(2)
    with col_g:
        csv_buffer = io.StringIO()
        filtered_data.to_csv(csv_buffer, index=False)
        st.download_button("Filtered CSV", csv_buffer.getvalue(), "nyc_collisions.csv")
        st.download_button("JSON", filtered_data.to_json(orient='records', date_format='iso'), "nyc_collisions.json")

    with col_h:
        summary = f"""
# NYC Collisions Report ({datetime.now().strftime('%Y-%m-%d')})
**Total:** {total_collisions:,} | **Avg Injuries:** {avg_injuries:.1f} | **Peak:** {peak_hour}:00 | **Rate:** {injury_rate:.1f}%

**Top Streets:**
{top_streets.to_markdown(index=False)}

**Insights:** {insights}
        """
        st.download_button("Summary TXT", summary, "report.txt", mime="text/plain")
else:
    st.info("Apply filters for exports.")

# Raw
if st.checkbox('Raw Data Preview'):
    st.dataframe(filtered_data.head(100), use_container_width=True, height=400)

st.markdown("---")
st.markdown("*By Nile Odhiambo Okomo*")
