import folium
import webbrowser
import tempfile
import os

ft_to_m = 0.3048

# Function to create a map with labeled circles, different colors, and satellite view
def plot_labeled_circles(lat_lng_radius_label_list, map_center, zoom_start=10, opacity=0):
    # Create a folium map centered at the given location with satellite view
    my_map = folium.Map(location=map_center, zoom_start=zoom_start, tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri')

    # Define a list of colors for each circle
    colors = ['blue', 'green', 'red', 'purple']

    # Iterate over the list of circles to plot
    for i, (lat, lng, radius, label) in enumerate(lat_lng_radius_label_list):
        color = colors[i % len(colors)]  # Cycle through colors
        
        # Add a circle to the map
        folium.Circle(
            location=(lat, lng),
            radius=radius*ft_to_m,  # Radius in meters
            color=color,    # Assign a different color
            fill=True,
            fill_color=color,
            fill_opacity=opacity
        ).add_to(my_map)

        # Add a marker with a popup that shows the label by default
        folium.Marker(
            location=(lat, lng),
            popup=folium.Popup(f"<b>{label}</b><br>Radius: {radius} ft", max_width=300),
            icon=folium.Icon(color=color)
        ).add_to(my_map)

    # Create a temporary file to store the map's HTML content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        file_path = tmp_file.name
        my_map.save(file_path)

    # Open the map in the default web browser
    webbrowser.open('file://' + os.path.realpath(file_path))

    my_map.save("bama_2025.html")

# List of (latitude, longitude, radius, label) tuples
lat_lng_radius_label_list = [
    (34.90066, -86.61550, 4750, 'Grain Silo -- NASA Antenna'),
    (34.89039, -86.61532, 1, '2024 USLI Recovery'),
    (34.89463, -86.61693, 2500, 'Flight Line'),  
    (34.89637, -86.61602, 3200, 'RSO Stand')
]

# Define the center of the map and initial zoom level
map_center = (34.89463, -86.61693)  # Center of the U.S.
zoom_start = 15

# Call the function to plot labeled circles
plot_labeled_circles(lat_lng_radius_label_list, map_center, zoom_start, opacity=0.1)
