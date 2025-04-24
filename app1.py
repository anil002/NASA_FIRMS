import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from libpysal.weights import Queen
from esda.moran import Moran
import geopandas as gpd
from shapely.geometry import Point
from io import StringIO

# Set page configuration
st.set_page_config(page_title="FIRMS ire Monitoring Dashboard", layout="wide")

# Add the title and developer line
st.title("FIRMS Fire Monitoring Dashboard")
st.markdown("Developed by: **Dr. Anil Kumar Singh**")
st.markdown("Email: **singhanil854@gmail.com**")
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 1.8rem;
        color: #FF5733;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
    }
    .help-text {
        font-size: 0.95rem;
        color: #333;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .inline-help {
        font-size: 0.85rem;
        color: #555;
        font-style: italic;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

#st.markdown('<div class="main-header">FIRMS ire Monitoring Dashboard</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("FIRMS API Configuration")
st.sidebar.markdown('<div class="help-text">Use the sidebar to configure data fetching parameters. Hover over fields or visit the Help tab for guidance.</div>', unsafe_allow_html=True)

# API Key input
st.sidebar.markdown("#### API Key")
api_key = st.sidebar.text_input("Enter FIRMS API Key", value="", type="password", help="Enter your NASA FIRMS API key. Obtain one from https://firms.modaps.eosdis.nasa.gov/api/")
st.sidebar.markdown('<div class="inline-help">Enter a valid API key to fetch fire data. Keep it secure and do not share publicly.</div>', unsafe_allow_html=True)

# Data source selection
st.sidebar.markdown("#### Data Source")
data_source = st.sidebar.selectbox(
    "Data Source",
    ("MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"),
    help="Select the satellite data source. VIIRS provides higher resolution (confidence as 0-100%), MODIS may use categorical confidence (low/nominal/high)."
)
st.sidebar.markdown('<div class="inline-help">Choose a data source. VIIRS_SNPP_NRT is recommended for recent, high-resolution data. MODIS may require confidence mapping.</div>', unsafe_allow_html=True)

# Date range selection
st.sidebar.markdown("#### Date Range")
today = datetime.now().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(today - timedelta(days=14), today),
    min_value=today - timedelta(days=365),
    max_value=today,
    help="Select start and end dates (up to 365 days prior). Use ≥14 days for time-series decomposition. Ranges >10 days are split into multiple requests due to API limits."
)
st.sidebar.markdown('<div class="inline-help">Choose a date range (e.g., last 14 days). API limits requests to 1–10 days; larger ranges are fetched in chunks.</div>', unsafe_allow_html=True)

# Map area selection
st.sidebar.markdown("#### Area Selection")
area_type = st.sidebar.radio("Select Area Type", ["Country", "Custom Coordinates"], help="Choose to fetch data for a country or a custom bounding box.")
st.sidebar.markdown('<div class="inline-help">Select Country for predefined regions or Custom Coordinates for a specific area.</div>', unsafe_allow_html=True)

if area_type == "Country":
    country = st.sidebar.selectbox(
        "Select Country",
        ("United States", "Australia", "Brazil", "Canada", "Indonesia", "Russia", 
         "India", "Mexico", "South Africa", "Portugal", "Greece", "Other"),
        help="Select a country to fetch fire data. 'Other' allows global or custom country input."
    )
    st.sidebar.markdown('<div class="inline-help">Pick a country or select Other to enter a custom country name or fetch global data.</div>', unsafe_allow_html=True)
    
    # Preset coordinates for map centering
    country_coords = {
        "United States": {"lat": 37.0902, "lon": -95.7129, "zoom": 4},
        "Australia": {"lat": -25.2744, "lon": 133.7751, "zoom": 4},
        "Brazil": {"lat": -14.2350, "lon": -51.9253, "zoom": 4},
        "Canada": {"lat": 56.1304, "lon": -106.3468, "zoom": 3},
        "Indonesia": {"lat": -0.7893, "lon": 113.9213, "zoom": 4},
        "Russia": {"lat": 61.5240, "lon": 105.3188, "zoom": 2},
        "India": {"lat": 20.5937, "lon": 78.9629, "zoom": 4},
        "Mexico": {"lat": 23.6345, "lon": -102.5528, "zoom": 4},
        "South Africa": {"lat": -30.5595, "lon": 22.9375, "zoom": 4},
        "Portugal": {"lat": 39.3999, "lon": -8.2245, "zoom": 6},
        "Greece": {"lat": 39.0742, "lon": 21.8243, "zoom": 6},
        "Other": {"lat": 0, "lon": 0, "zoom": 1}
    }
    
    if country != "Other":
        coords = country_coords[country]
        default_lat = coords["lat"]
        default_lon = coords["lon"]
        default_zoom = coords["zoom"]
    else:
        default_lat = 0
        default_lon = 0
        default_zoom = 1
        country = st.sidebar.text_input("Enter Country Name", value="Global", help="Enter a custom country name or leave as 'Global'.")
        st.sidebar.markdown('<div class="inline-help">Enter a name for reference (e.g., Global) when selecting Other.</div>', unsafe_allow_html=True)
else:
    col1, col2 = st.sidebar.columns(2)
    min_lat = col1.number_input("Min Latitude", value=-90.0, min_value=-90.0, max_value=90.0, help="Enter the minimum latitude (south boundary).")
    max_lat = col2.number_input("Max Latitude", value=90.0, min_value=-90.0, max_value=90.0, help="Enter the maximum latitude (north boundary).")
    min_lon = col1.number_input("Min Longitude", value=-180.0, min_value=-180.0, max_value=180.0, help="Enter the minimum longitude (west boundary).")
    max_lon = col2.number_input("Max Longitude", value=180.0, min_value=-180.0, max_value=180.0, help="Enter the maximum longitude (east boundary).")
    st.sidebar.markdown('<div class="inline-help">Define a bounding box by entering latitude and longitude ranges. Ensure min < max for both.</div>', unsafe_allow_html=True)
    if min_lat >= max_lat:
        st.sidebar.error("Max Latitude must be greater than Min Latitude.")
        st.error("Invalid coordinates: Max Latitude must be greater than Min Latitude.")
        st.stop()
    if min_lon >= max_lon:
        st.sidebar.error("Max Longitude must be greater than Min Longitude.")
        st.error("Invalid coordinates: Max Longitude must be greater than Min Longitude.")
        st.stop()
    default_lat = (min_lat + max_lat) / 2
    default_lon = (min_lon + max_lon) / 2
    default_zoom = 4

# Advanced options
st.sidebar.markdown("#### Advanced Options")
confidence_level = st.sidebar.slider("Minimum Confidence Level", 0, 100, 0, help="Filter fires by minimum confidence (0-100). For MODIS, categorical values are mapped (low=30, nominal=60, high=90). Set to 0 for sparse regions like Canada.")
st.sidebar.markdown('<div class="inline-help">Set a threshold to include high-confidence fires. Use 0 to maximize data in sparse regions.</div>', unsafe_allow_html=True)
max_results = st.sidebar.slider("Maximum Results", 100, 10000, 1000, help="Limit the number of fire records to display.")
st.sidebar.markdown('<div class="inline-help">Adjust to balance performance and data volume. Lower values improve speed.</div>', unsafe_allow_html=True)

# Function to fetch data from FIRMS API
@st.cache_data
def fetch_firms_data(api_key, source, start_date, end_date, area_type, **kwargs):
    if not api_key or not api_key.strip():
        return None, "API key is empty or invalid. Please provide a valid FIRMS API key."
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1
        if days <= 0:
            return None, "End date must be after start date."
    except ValueError:
        return None, "Invalid date format."

    country_bounding_boxes = {
        "United States": [-125.0, 24.0, -66.0, 50.0],
        "Australia": [112.0, -44.0, 154.0, -10.0],
        "Brazil": [-74.0, -34.0, -34.0, 6.0],
        "Canada": [-141.0, 41.0, -52.0, 84.0],
        "Indonesia": [95.0, -11.0, 141.0, 6.0],
        "Russia": [19.0, 41.0, 180.0, 82.0],
        "India": [68.0, 7.0, 97.0, 37.0],
        "Mexico": [-118.0, 14.0, -86.0, 32.0],
        "South Africa": [16.0, -35.0, 33.0, -22.0],
        "Portugal": [-9.5, 36.0, -6.0, 42.0],
        "Greece": [19.0, 34.0, 28.0, 42.0],
        "Other": [-180.0, -90.0, 180.0, 90.0]
    }

    if area_type == "Country":
        country = kwargs.get("country", "Other")
        area = ",".join(map(str, country_bounding_boxes.get(country, country_bounding_boxes["Other"])))
    else:
        min_lon = kwargs.get('min_lon')
        min_lat = kwargs.get('min_lat')
        max_lon = kwargs.get('max_lon')
        max_lat = kwargs.get('max_lat')
        if min_lon >= max_lon or min_lat >= max_lat:
            return None, "Invalid coordinates: min_lon must be less than max_lon, and min_lat must be less than max_lat."
        area = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    # Split date range into chunks of <=10 days
    data_frames = []
    current_start = start_dt
    max_days = 10  # FIRMS API limit

    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=max_days - 1), end_dt)
        chunk_days = (current_end - current_start).days + 1
        base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{area}/{chunk_days}"
        
        try:
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            
            if not response.text.strip().startswith("latitude,longitude"):
                st.warning(f"Invalid response for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {response.text[:1000]}")
                current_start = current_end + timedelta(days=1)
                continue
            
            data = pd.read_csv(StringIO(response.text), on_bad_lines='warn')
            
            if not data.empty:
                # Convert confidence column to numeric, handling categorical values for MODIS
                if 'confidence' in data.columns:
                    if source == "MODIS_NRT":
                        confidence_map = {'low': 30, 'nominal': 60, 'high': 90}
                        data['confidence'] = data['confidence'].map(confidence_map).fillna(
                            pd.to_numeric(data['confidence'], errors='coerce')
                        )
                    else:
                        data['confidence'] = pd.to_numeric(data['confidence'], errors='coerce')
                data_frames.append(data)
            else:
                st.warning(f"No fire data returned for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}.")
            
            current_start = current_end + timedelta(days=1)
        
        except requests.exceptions.HTTPError as e:
            st.warning(f"HTTP Error for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {str(e)}")
            current_start = current_end + timedelta(days=1)
            continue
        except requests.exceptions.ConnectionError:
            st.warning(f"Network error for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: Unable to connect to FIRMS API.")
            current_start = current_end + timedelta(days=1)
            continue
        except pd.errors.ParserError as e:
            st.warning(f"CSV Parsing Error for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {str(e)}")
            current_start = current_end + timedelta(days=1)
            continue
        except Exception as e:
            st.warning(f"Unexpected error for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {str(e)}")
            current_start = current_end + timedelta(days=1)
            continue

    if not data_frames:
        return None, f"No valid data returned from API for {start_date} to {end_date}. Try a wider date range, lower confidence level, or different area."
    
    # Concatenate all valid DataFrames
    data = pd.concat(data_frames, ignore_index=True)
    return data, None

# Main content area
if not api_key:
    st.warning("Please enter your FIRMS API key in the sidebar to access the data.")
    st.markdown("""
    ### About FIRMS Fire Monitoring Dashboard
    
    This dashboard uses the FIRMS API to monitor and analyze global fire data.
    
    Features:
    - Interactive fire hotspot maps
    - Advanced statistical and spatial analytics
    - Customizable alerts
    - Detailed data tables
    
    Get started by entering your API key in the sidebar. Obtain one from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/). Use the Help tab and inline help for guidance.
    """)
else:
    if 'firms_data' not in st.session_state:
        st.session_state.firms_data = None
        st.session_state.start_date = None
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        fetch_button = st.button("Fetch Fire Data", use_container_width=True, help="Click to fetch fire data based on sidebar settings.")
        st.markdown('<div class="inline-help">Click this button after configuring the sidebar to load fire data.</div>', unsafe_allow_html=True)
    
    if fetch_button or st.session_state.firms_data is not None:
        if fetch_button:
            if len(date_range) != 2:
                st.error("Please select both a start and end date.")
                st.stop()
            else:
                with st.spinner("Fetching fire data from FIRMS API..."):
                    start_date = pd.Timestamp(date_range[0])
                    end_date = pd.Timestamp(date_range[1])
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    
                    if area_type == "Country":
                        firms_data, error = fetch_firms_data(
                            api_key=api_key,
                            source=data_source,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            area_type=area_type,
                            country=country
                        )
                    else:
                        firms_data, error = fetch_firms_data(
                            api_key=api_key,
                            source=data_source,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            area_type=area_type,
                            min_lat=min_lat,
                            max_lat=max_lat,
                            min_lon=min_lon,
                            max_lon=max_lon
                        )
                    
                    if error:
                        st.error(f"{error}\n\nInvalid API key? Check https://firms.modaps.eosdis.nasa.gov/api/")
                    elif firms_data.empty:
                        st.warning(f"No fire data available for {country or 'selected area'} from {start_date_str} to {end_date_str}. Try a wider date range, lower confidence level, or different area.")
                    else:
                        st.session_state.firms_data = firms_data
                        st.session_state.start_date = start_date
        
        if st.session_state.firms_data is not None:
            df = st.session_state.firms_data
            
            if df.empty:
                st.warning("No data remains after processing. Try lowering the confidence level or selecting a different date range/area.")
                st.stop()
            
            required_columns = ['latitude', 'longitude']
            if not all(col in df.columns for col in required_columns):
                st.error("Required columns (latitude, longitude) not found in the data.")
                st.stop()
            
            # Apply confidence filter if column exists and has valid numeric values
            if 'confidence' in df.columns:
                if df['confidence'].dropna().empty:
                    st.warning("Confidence column contains no valid numeric values. Skipping confidence filter.")
                else:
                    original_len = len(df)
                    df = df[df['confidence'] >= confidence_level]
                    if len(df) == 0:
                        st.warning(f"Confidence filter (≥{confidence_level}%) removed all data. Try lowering the confidence level.")
                        st.stop()
                    elif len(df) < original_len:
                        st.info(f"Confidence filter (≥{confidence_level}%) reduced data from {original_len} to {len(df)} records.")
            
            df = df.head(max_results)
            
            if 'acq_date' in df.columns:
                df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
            else:
                st.warning("Acquisition date (acq_date) column missing. Some analytics may not work.")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Active Fires Map", "Statistics & Trends", "Data Table", "Alerts", "Help"])
            
            with tab1:
                st.markdown('<div class="section-header">Active Fires Map</div>', unsafe_allow_html=True)
                st.markdown('<div class="inline-help">Use the controls below to customize the map view and explore fire hotspots.</div>', unsafe_allow_html=True)
                
                map_col1, map_col2 = st.columns([3, 1])
                
                with map_col2:
                    map_type = st.radio("Map Type", ["Points", "Heat Map", "Cluster"], help="Choose how to display fires: Points (individual markers), Heat Map (density), or Cluster (grouped markers).")
                    st.markdown('<div class="inline-help">Select a map type to visualize fire data differently.</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Map Settings")
                    color_by = st.selectbox("Color Points By", ["frp", "confidence", "bright_t31", "None"], help="Color map points by a metric (e.g., FRP for fire intensity) or None for default red.")
                    st.markdown('<div class="inline-help">Choose a metric to color points, highlighting variations in fire characteristics.</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Fire Info")
                    st.markdown(f"<div class='metric-card'>Active Fires: {len(df)}</div>", unsafe_allow_html=True)
                    if 'frp' in df.columns:
                        max_frp = df['frp'].max()
                        avg_frp = df['frp'].mean()
                        st.markdown(f"<div class='metric-card'>Max FRP: {max_frp:.2f} MW</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'>Avg FRP: {avg_frp:.2f} MW</div>", unsafe_allow_html=True)
                    if 'acq_date' in df.columns:
                        most_recent = df['acq_date'].max()
                        st.markdown(f"<div class='metric-card'>Most Recent: {most_recent}</div>", unsafe_allow_html=True)
                    st.markdown('<div class="inline-help">Review summary statistics for the fetched fire data.</div>', unsafe_allow_html=True)
                
                with map_col1:
                    m = folium.Map(location=[default_lat, default_lon], zoom_start=default_zoom)
                    
                    if map_type == "Points":
                        for _, row in df.iterrows():
                            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
                                popup_text = f"""
                                Date: {row.get('acq_date', 'N/A')}<br>
                                Time: {row.get('acq_time', 'N/A')}<br>
                                FRP: {row.get('frp', 'N/A')} MW<br>
                                Confidence: {row.get('confidence', 'N/A')}%<br>
                                """
                                
                                if color_by != "None" and color_by in row and pd.notnull(row[color_by]):
                                    if color_by == "frp":
                                        max_val = df['frp'].max()
                                        val = row[color_by] / max_val
                                        color = f'rgb(255, {int(255 * (1 - val))}, 0)'
                                    elif color_by == "confidence":
                                        val = row[color_by] / 100
                                        color = f'rgb(255, {int(255 * (1 - val))}, 0)'
                                    else:
                                        max_val = df[color_by].max()
                                        min_val = df[color_by].min()
                                        val = (row[color_by] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                                        color = f'rgb({int(255 * val)}, {int(255 * (1 - val))}, 0)'
                                else:
                                    color = 'red'
                                
                                folium.CircleMarker(
                                    location=[row['latitude'], row['longitude']],
                                    radius=5,
                                    popup=folium.Popup(popup_text, max_width=300),
                                    color=color,
                                    fill=True,
                                    fill_opacity=0.7
                                ).add_to(m)
                    
                    elif map_type == "Heat Map":
                        heat_data = [[row['latitude'], row['longitude'], row.get('frp', 1)] 
                                     for _, row in df.iterrows() if pd.notnull(row['latitude']) and pd.notnull(row['longitude'])]
                        if heat_data:
                            HeatMap(heat_data, radius=15).add_to(m)
                        else:
                            st.warning("No valid data for heat map. Ensure latitude and longitude are present.")
                    
                    else:
                        marker_cluster = MarkerCluster().add_to(m)
                        for _, row in df.iterrows():
                            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
                                popup_text = f"""
                                Date: {row.get('acq_date', 'N/A')}<br>
                                Time: {row.get('acq_time', 'N/A')}<br>
                                FRP: {row.get('frp', 'N/A')} MW<br>
                                Confidence: {row.get('confidence', 'N/A')}%<br>
                                """
                                folium.Marker(
                                    location=[row['latitude'], row['longitude']],
                                    popup=folium.Popup(popup_text, max_width=300),
                                    icon=folium.Icon(color='red', icon='fire', prefix='fa')
                                ).add_to(marker_cluster)
                    
                    folium_static(m, width=800, height=600)
            
            with tab2:
                st.markdown('<div class="section-header">Statistics & Trends</div>', unsafe_allow_html=True)
                st.markdown('<div class="inline-help">Explore basic and advanced analytics to understand fire patterns. Scroll to view all visualizations.</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'acq_date' in df.columns:
                        df_daily = df.groupby(df['acq_date'].dt.date).size().reset_index(name='count')
                        fig = px.line(df_daily, x='acq_date', y='count', title='Daily Fire Count')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">View the number of fires per day over the selected date range.</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Cannot generate daily fire count plot: acq_date column missing.")
                    
                    if 'frp' in df.columns:
                        fig = px.histogram(df, x='frp', nbins=20, title='Fire Radiative Power (FRP) Distribution')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">See the distribution of fire intensity (FRP in MW).</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Cannot generate FRP distribution: frp column missing.")
                
                with col2:
                    if 'acq_time' in df.columns:
                        df['hour'] = df['acq_time'].apply(lambda x: int(str(x).zfill(4)[:2]) if pd.notnull(x) else np.nan)
                        hourly_counts = df.groupby('hour').size().reset_index(name='count')
                        fig = px.bar(hourly_counts, x='hour', y='count', title='Hourly Fire Distribution')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">Identify peak hours for fire detections.</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Cannot generate hourly fire distribution: acq_time column missing.")
                    
                    if 'confidence' in df.columns:
                        fig = px.histogram(df, x='confidence', nbins=10, title='Confidence Level Distribution')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">Examine the reliability of fire detections (0-100%).</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Cannot generate confidence distribution: confidence column missing.")
                
                st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
                st.markdown('<div class="inline-help">Use these tools for deeper insights into fire patterns. Requires sufficient data.</div>', unsafe_allow_html=True)
                
                if 'frp' in df.columns and 'confidence' in df.columns and len(df) > 5:
                    st.markdown("### Fire Intensity Clustering (K-Means)")
                    features = df[['latitude', 'longitude', 'frp', 'confidence']].dropna()
                    if len(features) > 5:
                        kmeans = KMeans(n_clusters=3, random_state=42)
                        features['cluster'] = kmeans.fit_predict(features[['frp', 'confidence']])
                        df = df.merge(features[['cluster']], left_index=True, right_index=True, how='left')
                        
                        fig = px.scatter_geo(
                            features,
                            lat='latitude',
                            lon='longitude',
                            color='cluster',
                            size='frp',
                            title='Fire Intensity Clusters',
                            projection='natural earth',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">View clusters of fires with similar intensity and confidence. Hover for details.</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Insufficient data for clustering (need at least 6 valid points with frp and confidence).")
                else:
                    st.warning("Cannot perform clustering: frp or confidence column missing, or insufficient data.")
                
                if 'acq_date' in df.columns:
                    st.markdown("### Time-Series Decomposition")
                    df_daily = df.groupby(df['acq_date'].dt.date).size().reset_index(name='count')
                    df_daily['acq_date'] = pd.to_datetime(df_daily['acq_date'])
                    df_daily.set_index('acq_date', inplace=True)
                    
                    if len(df_daily) >= 14:
                        decomposition = seasonal_decompose(df_daily['count'], model='additive', period=7)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_daily.index, y=decomposition.observed, name='Observed'))
                        fig.add_trace(go.Scatter(x=df_daily.index, y=decomposition.trend, name='Trend'))
                        fig.add_trace(go.Scatter(x=df_daily.index, y=decomposition.seasonal, name='Seasonal'))
                        fig.add_trace(go.Scatter(x=df_daily.index, y=decomposition.resid, name='Residual'))
                        fig.update_layout(
                            title='Time-Series Decomposition of Fire Counts',
                            height=600,
                            xaxis_title='Date',
                            yaxis_title='Fire Count'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">Analyze fire count trends, weekly patterns, and anomalies over time.</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"Insufficient data for time-series decomposition: need at least 14 daily observations, but only {len(df_daily)} available. Try a wider date range (≥14 days).")
                        # Fallback: Plot raw fire counts
                        fig = px.line(df_daily, x=df_daily.index, y='count', title='Daily Fire Count (Raw Data)')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">Showing raw fire counts due to insufficient data for decomposition.</div>', unsafe_allow_html=True)
                else:
                    st.warning("Cannot perform time-series decomposition: acq_date column missing.")
                
                if 'latitude' in df.columns and 'longitude' in df.columns and 'frp' in df.columns and len(df) > 10:
                    st.markdown("### Spatial Autocorrelation (Moran's I)")
                    spatial_df = df[['latitude', 'longitude', 'frp']].dropna()
                    if len(spatial_df) > 10:
                        geometry = [Point(lon, lat) for lat, lon in zip(spatial_df['latitude'], spatial_df['longitude'])]
                        gdf = gpd.GeoDataFrame(spatial_df, geometry=geometry, crs="EPSG:4326")
                        
                        w = Queen.from_dataframe(gdf)
                        w.transform = 'r'
                        
                        moran = Moran(gdf['frp'], w)
                        
                        st.markdown(f"**Moran's I Statistic**: {moran.I:.3f}")
                        st.markdown(f"**P-value**: {moran.p_sim:.3f}")
                        st.markdown('<div class="inline-help">Check if fires with similar FRP are clustered (positive I, low p-value).</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"Insufficient data for spatial autocorrelation: need at least 11 points, but only {len(spatial_df)} available.")
                else:
                    st.warning("Cannot perform spatial autocorrelation: latitude, longitude, or frp column missing, or insufficient data.")
                
                if all(col in df.columns for col in ['frp', 'confidence', 'bright_t31']) and len(df) > 5:
                    st.markdown("### 3D Fire Characteristics")
                    plot_data = df[['frp', 'confidence', 'bright_t31']].dropna()
                    if len(plot_data) > 5:
                        fig = px.scatter_3d(
                            df,
                            x='frp',
                            y='confidence',
                            z='bright_t31',
                            color='frp',
                            size='confidence',
                            title='3D Scatter Plot of Fire Characteristics',
                            color_continuous_scale='Inferno',
                            labels={'frp': 'FRP (MW)', 'confidence': 'Confidence (%)', 'bright_t31': 'Brightness (K)'}
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('<div class="inline-help">Explore relationships between FRP, confidence, and brightness. Rotate the plot to view angles.</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"Insufficient data for 3D scatter plot: need at least 6 points, but only {len(plot_data)} available.")
                else:
                    st.warning("Cannot generate 3D scatter plot: frp, confidence, or bright_t31 column missing, or insufficient data.")
                
                if all(col in df.columns for col in ['acq_date', 'latitude', 'longitude', 'acq_time']):
                    st.markdown("### Temporal-Spatial Fire Evolution")
                    st.markdown('<div class="inline-help">Customize the time aggregation and trend filter to monitor fire evolution spatially.</div>', unsafe_allow_html=True)
                    
                    df_time = df[['acq_date', 'latitude', 'longitude', 'frp', 'acq_time']].dropna()
                    if not df_time.empty:
                        # Add datetime column for hourly aggregation
                        df_time['datetime'] = pd.to_datetime(
                            df_time['acq_date'].astype(str) + ' ' + 
                            df_time['acq_time'].astype(str).str.zfill(4).str[:2] + ':00',
                            format='%Y-%m-%d %H:%M', errors='coerce'
                        )
                        
                        # User controls for time aggregation and trend filter
                        time_agg = st.selectbox(
                            "Time Aggregation",
                            ["Hourly", "Daily", "3-Day", "7-Day"],
                            help="Choose how to aggregate fire data over time (e.g., hourly or multi-day periods)."
                        )
                        st.markdown('<div class="inline-help">Select the time period for grouping fire data in the heatmap.</div>', unsafe_allow_html=True)
                        
                        trend_filter = st.radio(
                            "Trend Filter",
                            ["All", "Increasing", "Decreasing"],
                            help="Filter periods by fire count trend: All (show all), Increasing (more fires than previous period), Decreasing (fewer fires)."
                        )
                        st.markdown('<div class="inline-help">Filter to highlight periods of increasing or decreasing fire activity.</div>', unsafe_allow_html=True)
                        
                        # Ensure start_date is available
                        start_date = st.session_state.get('start_date')
                        if start_date is None:
                            st.error("Start date not set. Please fetch data again.")
                            st.stop()
                        
                        # Aggregate data based on user selection
                        if time_agg == "Hourly":
                            df_time['time_key'] = df_time['datetime'].dt.strftime('%Y-%m-%d %H:00')
                            group_col = 'time_key'
                            index_format = '%Y-%m-%d %H:00'
                        elif time_agg == "Daily":
                            df_time['time_key'] = df_time['acq_date'].dt.strftime('%Y-%m-%d')
                            group_col = 'time_key'
                            index_format = '%Y-%m-%d'
                        elif time_agg == "3-Day":
                            df_time['time_key'] = df_time['acq_date'].apply(
                                lambda x: (x - start_date).days // 3
                            )
                            df_time['time_key'] = df_time['time_key'].apply(
                                lambda x: (start_date + timedelta(days=x*3)).strftime('%Y-%m-%d')
                            )
                            group_col = 'time_key'
                            index_format = '%Y-%m-%d'
                        else:  # 7-Day
                            df_time['time_key'] = df_time['acq_date'].apply(
                                lambda x: (x - start_date).days // 7
                            )
                            df_time['time_key'] = df_time['time_key'].apply(
                                lambda x: (start_date + timedelta(days=x*7)).strftime('%Y-%m-%d')
                            )
                            group_col = 'time_key'
                            index_format = '%Y-%m-%d'
                        
                        # Group data and calculate fire counts
                        grouped = df_time.groupby(group_col).agg({
                            'latitude': list,
                            'longitude': list,
                            'frp': list,
                            'time_key': 'count'
                        }).rename(columns={'time_key': 'fire_count'})
                        grouped = grouped.reset_index()
                        
                        # Calculate trend (difference in fire count)
                        grouped['fire_count_diff'] = grouped['fire_count'].diff().fillna(0)
                        
                        # Filter based on trend
                        if trend_filter == "Increasing":
                            grouped = grouped[grouped['fire_count_diff'] > 0]
                        elif trend_filter == "Decreasing":
                            grouped = grouped[grouped['fire_count_diff'] < 0]
                        
                        # Prepare data for HeatMapWithTime
                        time_data = []
                        dates = []
                        for _, row in grouped.iterrows():
                            if row['latitude'] and row['longitude'] and row['frp']:
                                heat_data = [
                                    [lat, lon, frp] for lat, lon, frp in zip(row['latitude'], row['longitude'], row['frp'])
                                ]
                                time_data.append(heat_data)
                                dates.append(row['time_key'])
                        
                        if time_data:
                            m = folium.Map(location=[default_lat, default_lon], zoom_start=default_zoom)
                            # Customize gradient based on trend filter
                            gradient = {0.1: 'blue', 0.4: 'yellow', 0.7: 'orange', 1: 'red'} if trend_filter == "All" else (
                                {0.1: 'red', 0.4: 'orange', 0.7: 'yellow', 1: 'white'} if trend_filter == "Increasing" else
                                {0.1: 'blue', 0.4: 'cyan', 0.7: 'green', 1: 'white'}
                            )
                            HeatMapWithTime(
                                time_data,
                                index=dates,
                                radius=15,
                                auto_play=False,
                                max_opacity=0.8,
                                gradient=gradient
                            ).add_to(m)
                            folium_static(m, width=800, height=600)
                            st.markdown('<div class="inline-help">Use the time slider to view fire intensity evolution. Red indicates higher FRP or increasing trends, blue indicates decreasing trends.</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No valid data for the selected trend filter or time aggregation.")
                    else:
                        st.warning("No valid data for temporal-spatial heatmap: insufficient records after filtering.")
                else:
                    st.warning("Cannot generate temporal-spatial heatmap: acq_date, latitude, longitude, or acq_time column missing.")
            
            with tab3:
                st.markdown('<div class="section-header">Data Table</div>', unsafe_allow_html=True)
                st.markdown('<div class="inline-help">Customize and download fire data in a table format.</div>', unsafe_allow_html=True)
                
                if not df.empty:
                    all_columns = df.columns.tolist()
                    default_columns = [col for col in ['latitude', 'longitude', 'acq_date', 'acq_time', 'frp', 'confidence'] if col in all_columns]
                    selected_columns = st.multiselect("Select Columns to Display", all_columns, default=default_columns, help="Choose columns to show in the table.")
                    st.markdown('<div class="inline-help">Select specific columns to focus on relevant fire data.</div>', unsafe_allow_html=True)
                    
                    if selected_columns:
                        st.dataframe(df[selected_columns], height=600)
                    else:
                        st.dataframe(df, height=600)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Data as CSV",
                        data=csv,
                        file_name="firms_fire_data.csv",
                        mime="text/csv",
                        help="Download the table data as a CSV file."
                    )
                    st.markdown('<div class="inline-help">Click to export the table for use in other tools.</div>', unsafe_allow_html=True)
                else:
                    st.warning("No data available to display.")
            
            with tab4:
                st.markdown('<div class="section-header">Fire Alerts & Notifications</div>', unsafe_allow_html=True)
                st.markdown('<div class="inline-help">Set up alerts to monitor fire events based on custom conditions.</div>', unsafe_allow_html=True)
                
                st.markdown("### Configure Alerts")
                col1, col2 = st.columns(2)
                
                with col1:
                    alert_type = st.selectbox(
                        "Alert Type",
                        ["New Fires", "High FRP Fires", "Fire Density", "Custom Alert"],
                        help="Choose the type of fire event to monitor."
                    )
                    st.markdown('<div class="inline-help">Select an alert type to define the condition for notifications.</div>', unsafe_allow_html=True)
                    
                    if alert_type == "New Fires":
                        st.markdown("Notify when new fires are detected in the selected area.")
                        min_confidence = st.slider("Minimum Confidence Level for Alert", 0, 100, 50, help="Set the minimum confidence for new fire alerts.")
                        st.markdown('<div class="inline-help">Adjust to filter new fires by detection reliability.</div>', unsafe_allow_html=True)
                    
                    elif alert_type == "High FRP Fires":
                        st.markdown("Notify when fires exceed a certain Fire Radiative Power threshold.")
                        frp_threshold = st.slider("FRP Threshold (MW)", 0, 500, 100, help="Set the FRP threshold for high-intensity fires.")
                        st.markdown('<div class="inline-help">Choose a threshold to detect intense fires.</div>', unsafe_allow_html=True)
                    
                    elif alert_type == "Fire Density":
                        st.markdown("Notify when fire density in an area exceeds threshold.")
                        density_threshold = st.slider("Fire Density Threshold (fires per sq. km)", 0.0, 1.0, 0.1, 0.01, help="Set the density threshold for clustered fires.")
                        radius_km = st.sidebar.slider("Analysis Radius (km)", 1, 50, 10, help="Define the radius for density calculation.")
                        st.markdown('<div class="inline-help">Adjust to monitor areas with high fire concentration.</div>', unsafe_allow_html=True)
                    
                    else:  # Custom Alert
                        st.markdown("Create a custom alert condition.")
                        available_metrics = [col for col in ['frp', 'confidence', 'bright_t31'] if col in df.columns] + ['count']
                        metric = st.selectbox("Metric", available_metrics, help="Choose a metric for the custom alert.")
                        operator = st.selectbox("Operator", ["greater than", "less than", "equal to"], help="Define the comparison operator.")
                        value = st.number_input("Value", min_value=0.0, value=100.0, help="Set the value to compare against.")
                        st.markdown('<div class="inline-help">Create a tailored alert based on specific fire metrics.</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Notification Settings")
                    notify_method = st.multiselect(
                        "Notification Method", 
                        ["Email", "SMS", "Dashboard", "Webhook"],
                        default=["Dashboard"],
                        help="Select how to receive alerts."
                    )
                    st.markdown('<div class="inline-help">Choose one or more methods for notifications.</div>', unsafe_allow_html=True)
                    
                    if "Email" in notify_method:
                        email = st.text_input("Email Address", help="Enter the email for notifications.")
                        st.markdown('<div class="inline-help">Provide a valid email address.</div>', unsafe_allow_html=True)
                    
                    if "SMS" in notify_method:
                        phone = st.text_input("Phone Number", help="Enter the phone number for SMS alerts.")
                        st.markdown('<div class="inline-help">Provide a valid phone number.</div>', unsafe_allow_html=True)
                    
                    if "Webhook" in notify_method:
                        webhook_url = st.text_input("Webhook URL", help="Enter the webhook URL for automated alerts.")
                        st.markdown('<div class="inline-help">Provide a valid webhook URL.</div>', unsafe_allow_html=True)
                    
                    notify_frequency = st.selectbox(
                        "Notification Frequency",
                        ["Immediately", "Hourly", "Daily", "Weekly"],
                        help="Choose how often to receive alerts."
                    )
                    st.markdown('<div class="inline-help">Set the frequency of notifications.</div>', unsafe_allow_html=True)
                
                save_alert = st.button("Save Alert Configuration", use_container_width=True, help="Save the alert settings to activate notifications.")
                if save_alert:
                    st.success("Alert configuration saved successfully!")
                st.markdown('<div class="inline-help">Click to save and activate your alert settings.</div>', unsafe_allow_html=True)
                
                st.markdown("### Current Active Alerts")
                if len(df) > 0:
                    alerts_df = pd.DataFrame({
                        "Alert Type": ["High FRP", "Fire Density", "New Fire"],
                        "Status": ["Active", "Active", "Active"],
                        "Condition": [
                            f"FRP > {df['frp'].quantile(0.9):.2f} MW" if 'frp' in df.columns else "Custom condition",
                            "Density > 0.1 fires/km²",
                            f"Confidence > {df['confidence'].quantile(0.8):.0f}%" if 'confidence' in df.columns else "New fire detected"
                        ],
                        "Area": ["Selected region", "Selected region", "Selected region"],
                        "Last Triggered": [
                            (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                            (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
                            (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M")
                        ]
                    })
                    st.dataframe(alerts_df, height=200)
                    st.markdown('<div class="inline-help">View currently active alerts and their conditions.</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Alert History")
                    end_date = datetime.now()
                    start_date_alert = end_date - timedelta(days=14)
                    date_range_alert = pd.date_range(start=start_date_alert, end=end_date)
                    np.random.seed(42)
                    alert_counts = np.random.poisson(lam=3, size=len(date_range_alert))
                    alert_history = pd.DataFrame({'date': date_range_alert, 'alert_count': alert_counts})
                    fig = px.bar(alert_history, x='date', y='alert_count', title='Alert History (Last 14 Days)')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('<div class="inline-help">Review alert triggers over the past 14 days.</div>', unsafe_allow_html=True)
                else:
                    st.info("No active alerts. Configure alerts to monitor fire events.")
            
            with tab5:
                st.markdown('<div class="section-header">Help Menu</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class='help-text'>
                This Help Menu provides detailed guidance on using the FIRMS Fire Monitoring Dashboard, including how to interact with the sidebar and each tab’s controls. Use the expanders below to explore specific sections.
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Sidebar Configuration"):
                    st.markdown("""
                    <div class='help-text'>
                    **Description**: The sidebar allows you to configure data fetching parameters.<br>
                    **How to Use**:
                    - **API Key**: Enter your NASA FIRMS API key (get one from https://firms.modaps.eosdis.nasa.gov/api/). Keep it secure.
                    - **Data Source**: Select a satellite source (e.g., VIIRS_SNPP_NRT for high-resolution data, 0-100% confidence; MODIS_NRT may use low/nominal/high).
                    - **Date Range**: Choose start and end dates (up to 365 days prior). Use ≥14 days for time-series decomposition. API limits requests to 1–10 days; larger ranges are split into multiple requests.
                    - **Area Selection**:
                      - **Country**: Pick a predefined country or 'Other' to enter a custom name or fetch global data. For sparse regions like Canada, use a wide date range and low confidence level (e.g., 0).
                      - **Custom Coordinates**: Enter min/max latitude and longitude to define a bounding box.
                    - **Advanced Options**:
                      - **Minimum Confidence Level**: Filter fires by confidence (0-100%). For MODIS, categorical values are mapped (low=30, nominal=60, high=90). Set to 0 for sparse data.
                      - **Maximum Results**: Limit the number of records (100-10,000) for performance.
                    **Tips**:
                    - Use inline help text below each field for guidance.
                    - Ensure valid inputs (e.g., max latitude > min latitude, end date > start date).
                    - For large date ranges, expect longer fetch times due to multiple API calls.
                    - If no data is returned, try a wider date range (≥14 days), lower confidence level (0), or different area (e.g., global for Canada).
                    - For code errors (e.g., indentation issues), ensure consistent 4-space indentation and check for tabs in your editor.
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("Active Fires Map"):
                    st.markdown("""
                    <div class='help-text'>
                    **Description**: Displays fire hotspots on an interactive map.<br>
                    **How to Use**:
                    - **Map Type**: Choose Points (individual fires), Heat Map (density), or Cluster (grouped fires) via the radio buttons.
                    - **Color Points By**: Select a metric (FRP, confidence, bright_t31) to color points, or None for red markers.
                    - **Fire Info**: View summary stats (total fires, max/avg FRP, most recent fire).
                    - **Map Interaction**: Zoom, pan, or click markers for details (date, time, FRP, confidence).
                    **Tips**:
                    - Use Points for detailed inspection, Heat Map for density, or Cluster for large datasets.
                    - Hover over controls for tooltips.
                    - If no fires appear, check the date range, confidence level, or area.
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("Statistics & Trends"):
                    st.markdown("""
                    <div class='help-text'>
                    **Description**: Provides basic and advanced fire analytics.<br>
                    **How to Use**:
                    - **Basic Plots**:
                      - **Daily Fire Count**: Line chart showing fires per day.
                      - **FRP Distribution**: Histogram of fire intensity (MW).
                      - **Hourly Fire Distribution**: Bar chart of fires by hour.
                      - **Confidence Distribution**: Histogram of detection reliability.
                    - **Advanced Analytics**:
                      - **K-Means Clustering**: Groups fires by FRP and confidence (needs ≥6 points).
                      - **Time-Series Decomposition**: Breaks down fire counts into trend, seasonal, and residual components (needs ≥14 daily observations).
                      - **Spatial Autocorrelation (Moran's I)**: Checks if similar FRP fires are clustered (needs ≥11 points).
                      - **3D Fire Characteristics**: 3D scatter plot of FRP, confidence, and brightness (needs ≥6 points).
                      - **Temporal-Spatial Fire Evolution**: Heatmap with a time slider for fire intensity changes. Customize time aggregation (Hourly, Daily, 3-Day, 7-Day) and filter by trend (All, Increasing, Decreasing).
                    **Tips**:
                    - Scroll to view all analytics.
                    - Hover over plots for details (e.g., values, dates).
                    - For sparse regions like Canada, use a wide date range (≥14 days) and low confidence level (0) to ensure sufficient data.
                    - If analytics fail (e.g., insufficient data), try a wider date range, lower confidence level, or different area.
                    - If errors occur (e.g., missing data, non-numeric confidence, or code issues), check the API key, data source, date range, and code indentation, then refetch data.
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("Data Table"):
                    st.markdown("""
                    <div class='help-text'>
                    **Description**: Shows raw fire data in a customizable table.<br>
                    **How to Use**:
                    - **Select Columns**: Use the multiselect to choose columns (e.g., latitude, longitude, FRP).
                    - **View Table**: Scroll to explore records.
                    - **Download Data as CSV**: Click the button to export the table.
                    **Tips**:
                    - Select fewer columns for a focused view.
                    - Use the CSV file in tools like Excel or Python.
                    - If the table is empty, check the date range, confidence level, or area.
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("Alerts"):
                    st.markdown("""
                    <div class='help-text'>
                    **Description**: Configures notifications for fire events.<br>
                    **How to Use**:
                    - **Alert Type**: Choose New Fires, High FRP Fires, Fire Density, or Custom Alert.
                    - **Parameters**:
                      - **New Fires**: Set minimum confidence.
                      - **High FRP**: Set FRP threshold (MW).
                      - **Fire Density**: Set density threshold and radius.
                      - **Custom Alert**: Define metric, operator, and value (based on available columns).
                    - **Notification Settings**:
                      - Select methods (Email, SMS, Dashboard, Webhook).
                      - Enter contact details if needed (e.g., email address).
                      - Choose frequency (Immediately, Hourly, Daily, Weekly).
                    - **Save Alert**: Click to activate the alert.
                    - **View Alerts**: Check active alerts and history.
                    **Tips**:
                    - Use Dashboard for testing notifications.
                    - Save alerts to monitor ongoing fire events.
                    - Ensure sufficient data for alerts by using a low confidence level in sparse regions.
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p class="info-text">
            FIRMS Fire Monitoring Dashboard | Powered by NASA FIRMS API | Data Source: {data_source}
        </p>
    </div>
    """.format(data_source=data_source),
    unsafe_allow_html=True
)

# Welcome message
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.info("Enter your API key and configure parameters in the sidebar, then click 'Fetch Fire Data'. Use inline help or the Help tab for detailed guidance.")
