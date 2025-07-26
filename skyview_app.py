# SkyView Enthusiast - A Streamlit Flight Tracker Application
#
# To run this application:
# 1. Save this code as a Python file (e.g., `skyview_app.py`).
# 2. Open your terminal or command prompt.
# 3. Ensure you have the latest libraries:
#    pip install --upgrade streamlit pandas requests geopy pydeck streamlit-aggrid
# 4. Run the app with the command:
#    streamlit run skyview_app.py

import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import math
import time
from collections import deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Configuration and Page Setup ---
st.set_page_config(
    page_title="SkyView Enthusiast",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'tracked_locations' not in st.session_state:
    st.session_state.tracked_locations = {}
if 'selected_icao' not in st.session_state:
    st.session_state.selected_icao = None

# --- Helper Functions ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points on Earth in miles."""
    R = 3958.8
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_triangle_polygon(lon, lat, heading, size_deg=0.015):
    """Creates the vertices for a triangle polygon to represent a plane."""
    angle_rad = math.radians(90 - heading)
    p1 = (lon + size_deg * math.cos(angle_rad), lat + size_deg * math.sin(angle_rad))
    p2 = (lon + size_deg * math.cos(angle_rad + math.radians(140)), lat + size_deg * math.sin(angle_rad + math.radians(140)))
    p3 = (lon + size_deg * math.cos(angle_rad - math.radians(140)), lat + size_deg * math.sin(angle_rad - math.radians(140)))
    return [[p1, p2, p3]]

def get_circle_points(lat, lon, radius_miles, num_points=100):
    """Generates points to draw a circle on the map."""
    points = []
    earth_radius_miles = 3958.8
    lat_rad = math.radians(lat)
    for i in range(num_points + 1):
        angle = i * (360 / num_points)
        angle_rad = math.radians(angle)
        lat_offset = (radius_miles / earth_radius_miles) * math.cos(angle_rad)
        lon_offset = (radius_miles / (earth_radius_miles * math.cos(lat_rad))) * math.sin(angle_rad)
        points.append([lon + math.degrees(lon_offset), lat + math.degrees(lat_offset)])
    return [points]


# --- Data Fetching and Caching ---

@st.cache_resource
def get_geocoder():
    """Initializes a Nominatim geocoder instance."""
    return Nominatim(user_agent="skyview_enthusiast_app")

@st.cache_data(ttl=3600)
def geocode_location(address):
    """Converts a string address/zip to latitude and longitude."""
    if not address: return None, None, "Invalid Address"
    try:
        location = get_geocoder().geocode(address, country_codes="US")
        if location: return location.latitude, location.longitude, location.address
        else: return None, None, f"Could not find location for: {address}"
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        st.error(f"Geocoding service is unavailable. Error: {e}")
        return None, None, "Geocoding service error."

@st.cache_data(show_spinner=False)
def get_aircraft_database():
    """Downloads and caches the OpenSky aircraft database."""
    url = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
    try:
        df = pd.read_csv(url)
        df = df[['icao24', 'manufacturername', 'model']]
        df = df.rename(columns={'manufacturername': 'manufacturer', 'model': 'model_type'})
        return df
    except Exception as e:
        st.warning(f"Could not load aircraft database. Error: {e}")
        return None

def get_bounding_box(lat, lon, radius_miles):
    """Calculates the bounding box for a given center point and radius."""
    earth_radius_miles = 3958.8
    lat_rad = math.radians(lat)
    lat_delta = radius_miles / earth_radius_miles
    lon_delta = radius_miles / (earth_radius_miles * math.cos(lat_rad))
    return (lat - math.degrees(lat_delta), lat + math.degrees(lat_delta),
            lon - math.degrees(lon_delta), lon + math.degrees(lon_delta))

@st.cache_data(ttl=30, show_spinner=False)
def get_flight_data(bounding_box):
    """Fetches flight data from the OpenSky Network API."""
    min_lat, max_lat, min_lon, max_lon = bounding_box
    url = f"https://opensky-network.org/api/states/all?lamin={min_lat}&lomin={min_lon}&lamax={max_lat}&lomax={max_lon}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('states', [])
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 429:
            st.error(f"API rate limit exceeded. Please wait a moment.")
        return None

# New function to get photo data from Planespotters.net API
@st.cache_data(ttl=3600, show_spinner=False)
def get_plane_photo_data(icao):
    """Fetches photo data for a given ICAO24 hex code."""
    if not icao:
        return None
    url = f"https://api.planespotters.net/pub/photos/hex/{icao}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('photos') and len(data['photos']) > 0:
            return data['photos'][0] # Return the first photo object
        return None
    except requests.exceptions.RequestException:
        return None

# --- Main Application UI ---

st.title("✈️ SkyView Enthusiast")
st.markdown("Track real-time air traffic over any location in the United States.")

MAX_RADIUS = 200

with st.sidebar:
    st.header("Search Controls")
    address_input = st.text_input("Enter US Address or ZIP Code", "Space Needle, Seattle, WA", help="e.g., Space Needle, Seattle or 90210")
    radius_input = st.slider("Filter Radius (miles)", min_value=10, max_value=MAX_RADIUS, value=50, step=10)
    
    st.header("Map Options")
    show_paths = st.toggle("Show Flight Paths", value=True)
    home_dot_size = st.slider("Home Location Dot Size", min_value=100, max_value=1000, value=200, step=50, help="Size in meters for the black dot on the map.")

    st.markdown("---")
    st.info("Data from The OpenSky Network, updated every 60 seconds.")
    st.caption("Click a row in the table to highlight a plane and see its photo.")


# --- Main Content Area ---
lat, lon, full_address = geocode_location(address_input)

if lat is None or lon is None:
    st.error(f"Could not determine location for '{address_input}'. Please enter a valid US address or ZIP code.")
else:
    st.success(f"📍 Centered on: **{full_address}**")

    if full_address not in st.session_state.tracked_locations:
        st.session_state.tracked_locations[full_address] = {'paths': {}}
        st.session_state.selected_icao = None # Reset selection on new address

    aircraft_db = get_aircraft_database()
    bbox = get_bounding_box(lat, lon, MAX_RADIUS)
    with st.spinner(f"Scanning for aircraft within {MAX_RADIUS} miles..."):
        flight_states = get_flight_data(bbox)

    if flight_states is None:
        st.warning("Could not retrieve flight data at this time.")
    
    column_names = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source']
    
    if flight_states:
        master_df = pd.DataFrame(flight_states, columns=column_names)
        master_df.dropna(subset=['longitude', 'latitude', 'true_track'], inplace=True)
        master_df['callsign'] = master_df['callsign'].str.strip()
        master_df['altitude_ft'] = master_df['baro_altitude'] * 3.28084
        master_df['velocity_knots'] = master_df['velocity'] * 1.94384
        master_df['distance_miles'] = master_df.apply(lambda row: haversine_distance(lat, lon, row['latitude'], row['longitude']), axis=1)
        
        if aircraft_db is not None:
            master_df = pd.merge(master_df, aircraft_db, on='icao24', how='left')
        
        master_df['aircraft_type'] = (master_df.get('manufacturer', pd.Series(index=master_df.index)).fillna('') + ' ' + master_df.get('model_type', pd.Series(index=master_df.index)).fillna('')).str.strip().replace('', 'N/A')
        master_df.fillna({'callsign': 'N/A', 'aircraft_type': 'N/A', 'altitude_ft': 0, 'velocity_knots': 0, 'origin_country': 'N/A'}, inplace=True)
        master_df = master_df.round({'altitude_ft': 0, 'velocity_knots': 1, 'true_track': 1, 'distance_miles': 2})

        current_location_paths = st.session_state.tracked_locations[full_address]['paths']
        for index, row in master_df.iterrows():
            icao = row['icao24']
            if icao not in current_location_paths:
                current_location_paths[icao] = deque(maxlen=500)
            current_pos = [row['longitude'], row['latitude']]
            if not list(current_location_paths[icao]) or current_location_paths[icao][-1] != current_pos:
                 current_location_paths[icao].append(current_pos)
        
        display_df = master_df[master_df['distance_miles'] <= radius_input].copy()
    else:
        display_df = pd.DataFrame(columns=column_names + ['distance_miles', 'aircraft_type'])

    # --- Map and Table Rendering ---
    map_container = st.container()
    image_container = st.container()
    table_container = st.container()

    if display_df.empty:
        with map_container:
            st.info(f"No aircraft detected within {radius_input} miles of {full_address}.")
    else:
        # --- Highlighting Logic ---
        colors = [[255, 0, 0, 255] for _ in range(len(display_df))]
        
        if st.session_state.selected_icao:
            try:
                loc_idx = display_df.index.get_loc(display_df[display_df['icao24'] == st.session_state.selected_icao].index[0])
                colors[loc_idx] = [255, 255, 0, 255]
            except (IndexError, KeyError):
                pass
        
        display_df['color'] = colors

        path_data = []
        if show_paths:
            visible_icaos = set(display_df['icao24'])
            current_location_paths = st.session_state.tracked_locations[full_address]['paths']
            for icao, path in current_location_paths.items():
                if icao in visible_icaos and len(path) > 1:
                     path_data.append({"path": list(path)})

        with map_container:
            st.subheader(f"📡 Found {len(display_df)} Aircraft within {radius_input} miles")
            
            home_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
            home_layer = pdk.Layer("ScatterplotLayer", data=home_df, get_position='[lon, lat]', get_fill_color=[0, 0, 0, 255], get_radius=home_dot_size, pickable=False)
            
            circle_points = get_circle_points(lat, lon, radius_input)
            radius_layer = pdk.Layer("PathLayer", data=[{"path": circle_points[0]}], get_path="path", get_width=2, get_color=[0, 0, 0, 150], width_min_pixels=2)

            display_df['polygon'] = display_df.apply(lambda row: get_triangle_polygon(row['longitude'], row['latitude'], row['true_track']), axis=1)
            plane_layer = pdk.Layer(
                "PolygonLayer",
                data=display_df,
                get_polygon="polygon",
                get_fill_color='color',
                pickable=True,
                auto_highlight=True,
                extruded=False,
            )

            path_layer = pdk.Layer("PathLayer", data=path_data, get_path="path", get_width=3, get_color=[255, 220, 0, 255], width_min_pixels=2)

            layers_to_render = [home_layer, radius_layer, plane_layer]
            if show_paths: layers_to_render.append(path_layer)

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/satellite-streets-v11',
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=8, pitch=0),
                layers=layers_to_render,
                tooltip={"html": "<b>Flight:</b> {callsign}<br/><b>Aircraft:</b> {aircraft_type}<br/><b>Distance:</b> {distance_miles} mi"}
            ))
        
        # --- Aircraft Image Display ---
        with image_container:
            if st.session_state.selected_icao:
                selected_plane_data = display_df[display_df['icao24'] == st.session_state.selected_icao]
                if not selected_plane_data.empty:
                    plane = selected_plane_data.iloc[0]
                    st.subheader(f"Photo of {plane['callsign'].strip()} - {plane['aircraft_type']}")
                    
                    # Call the new function to get photo data
                    photo_data = get_plane_photo_data(plane['icao24'])
                    
                    if photo_data:
                        image_url = photo_data['thumbnail_large']['src']
                        photographer = photo_data['photographer']
                        photo_link = photo_data['link']
                        
                        st.image(image_url, use_container_width=True)
                        st.caption(f"Photo by [{photographer}]({photo_link}) on Planespotters.net")
                    else:
                        st.info("Photo not available for this aircraft.")
                else:
                    st.info("Selected plane is no longer in view. Clear selection by clicking another plane.")
            else:
                st.info("Click a row in the table below to see a photo of the aircraft.")


        # --- Data Table Display ---
        with table_container:
            st.subheader("Live Flight Data Table")
            display_columns = {'icao24': 'icao24', 'callsign': 'Flight', 'aircraft_type': 'Aircraft Type', 'distance_miles': 'Distance (mi)', 'altitude_ft': 'Altitude (ft)', 'velocity_knots': 'Speed (knots)', 'true_track': 'Heading (°)', 'origin_country': 'Origin Country'}
            
            grid_df = display_df[list(display_columns.keys())].rename(columns=display_columns)

            gb = GridOptionsBuilder.from_dataframe(grid_df)
            gb.configure_selection(selection_mode="single", use_checkbox=False)
            gb.configure_column("Distance (mi)", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
            
            gridOptions = gb.build()

            ag_grid_response = AgGrid(
                grid_df,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                data_return_mode=DataReturnMode.AS_INPUT,
                allow_unsafe_jscode=True,
                height=300,
                width='100%',
                reload_data=True,
                key=f'flight_table_{full_address}'
            )

            selected_rows_df = pd.DataFrame(ag_grid_response['selected_rows'])
            
            new_selected_icao = None
            if not selected_rows_df.empty:
                new_selected_icao = selected_rows_df.iloc[0]['icao24']

            if st.session_state.selected_icao != new_selected_icao:
                st.session_state.selected_icao = new_selected_icao
                st.rerun()

    # Auto-refresh loop
    time.sleep(30)
    st.rerun()
