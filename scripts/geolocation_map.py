# Import and libraries 

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from PIL import Image

csv_file = '.../Raw_data/birdclef-2025/train.csv'
species_list = [
   "Elaenia flavogaster",       
    "Penelope purpurascens",       
    "Megarynchus pitangua", 
    "Andinobates opisthomelas",  
    "Pyrilia pyrilia",  
    "Panthera onca",  
    "Alouatta seniculus", 
    "Bradypus variegatus",  
    "Colostethus inguinalis",  
    "Cerdocyon thous",  
    "Allobates niputidea",  
    "Lontra longicaudis", 
    "Crax alberti" 
]

# Read and filter

df = pd.read_csv(csv_file)
df_filtered = df[df['scientific_name'].isin(species_list)]
grouped = df_filtered.groupby(['scientific_name', 'latitude', 'longitude']).size().reset_index(name='count')

# Reverse geocodin

geolocator = Nominatim(user_agent="geoapiExercises")
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

locations = []
for idx, row in grouped.iterrows():
    lat, lon = row['latitude'], row['longitude']
    try:
        location = geocode((lat, lon), language='es')
        address = location.address if location else "==== Location not found ====="
    except Exception as e:
        print(f"Error in ({lat}, {lon}): {e}")
        address = "Error"
    locations.append(address)

grouped['location'] = locations
grouped.to_csv("species_locations_with_place.csv", index=False)
print("==== [INFO] CSV saved: species_locations_with_place.csv =====")

#Create Map

mean_lat = grouped['latitude'].mean()
mean_lon = grouped['longitude'].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=4, tiles="CartoDB positron")

for _, row in grouped.iterrows():
    popup_text = (
        f"<b>Species:</b> {row['scientific_name']}<br>"
        f"<b>Recordings:</b> {row['count']}<br>"
        f"<b>Location:</b> {row['location']}"
    )
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

html_map_file = "species_map.html"
m.save(html_map_file)
print(f"==== [INFO] HTML map saved: {html_map_file} =====")

# Convert HTML to PNG and PDF
# Configuration headless
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=2400,1800")  # large size for high resolution

# Download the driver 

driver = webdriver.Chrome(executable_path="C:/Users/andre/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe", options=chrome_options)
driver.get(f"file://{html_map_file}")
time.sleep(5)  # wait load 5 seconds each query

png_file = "species_map.png"
driver.save_screenshot(png_file)
driver.quit()
print(f"==== PNG image saved: {png_file} =====")

# Abrir PNG y exportar a PDF en 800 DPI
img = Image.open(png_file)
pdf_file = "species_map.pdf"
img.save(pdf_file, "PDF", resolution=800.0)
print(f"==== PDF saved en alta resolución: {pdf_file} ====")
