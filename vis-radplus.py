import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import glob
    import json
    return glob, json


@app.cell
def _(glob, json):
    for file in glob.glob("radhackathon.daten/Radplus/*/*.json"):
        print(f"\n📄 {file}")
        with open(file) as f2:
            d = json.load(f2)
            # Convert JSON object to a formatted string
            s = json.dumps(d, indent=2)
            # Print only the first 500 characters
            print(s[:500] + ("..." if len(s) > 500 else ""))
    return


@app.cell
def _(json):
    def _():
        import folium
        from folium.plugins import MeasureControl

        # Load GeoJSON
        with open("radhackathon.daten/rad+potsdam/p-quelle-ziel-zellen/data.json") as f:
            geojson_data = json.load(f)

        #geojson_data["features"] = geojson_data["features"]
        # Create map
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)

        m.add_child(MeasureControl(
            primary_length_unit='kilometers',
            secondary_length_unit='meters',
            primary_area_unit='sqmeters'
        ))
        # Add GeoJSON layer
        folium.GeoJson(
            geojson_data,
            name="GeoJSON Data"
        ).add_to(m)

        # Display in Jupyter or export to HTML
        m.save("map.html")
        return m


    _()
    return


@app.cell
def _(folium, json):
    def _():
        # Load OD counts
        with open("radhackathon.daten/rad+potsdam/p-quelle-ziel-statistiken/data.json") as f:
            od_data = json.load(f)

        # Load H3 cell geometries
        with open("radhackathon.daten/rad+potsdam/p-quelle-ziel-zellen/data.json") as f:
            h3_cells = json.load(f)

        # Build a mapping: H3 index -> coordinates
        h3_coords = {feat["properties"]["h3"]: feat["geometry"]["coordinates"] for feat in h3_cells["features"]}

        # Initialize map (rough center of Berlin)
        m = folium.Map(location=[52.52, 13.405], zoom_start=12)

        # Plot OD flows as lines
        for entry in od_data:
            org = entry["org_h3"]
            dst = entry["dst_h3"]
            count = entry["count"]

            if org in h3_coords and dst in h3_coords:
                org_coord = h3_coords[org][::-1]  # Folium uses [lat, lon]
                dst_coord = h3_coords[dst][::-1]

                # Draw a line representing flow, width proportional to count
                folium.PolyLine(
                    locations=[org_coord, dst_coord],
                    color="blue",
                    weight=max(1, count/10),  # adjust scaling as needed
                    opacity=0.6,
                ).add_to(m)

        # Optional: add markers for all H3 cells
        for h3, coord in h3_coords.items():
            folium.CircleMarker(
                location=coord[::-1],
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.7,
                popup=f"H3: {h3}"
            ).add_to(m)

        # Save map
        m.save("radplus_od_map.html")
        return m


    _()
    return


@app.cell
def _(folium, json):
    def _():
        # Load OD counts
        with open("radhackathon.daten/rad+potsdam/p-quelle-ziel-statistiken/data.json") as f:
            od_data = json.load(f)

        # Load H3 cell geometries
        with open("radhackathon.daten/rad+potsdam/p-quelle-ziel-zellen/data.json") as f:
            h3_cells = json.load(f)

        # Build a mapping: H3 index -> coordinates
        h3_coords = {feat["properties"]["h3"]: feat["geometry"]["coordinates"] for feat in h3_cells["features"]}

        # Count total connections per H3 cell
        from collections import defaultdict
        conn_counts = defaultdict(int)
        for entry in od_data:
            conn_counts[entry["org_h3"]] += entry["count"]
            conn_counts[entry["dst_h3"]] += entry["count"]

        # Initialize map
        m = folium.Map(location=[52.52, 13.405], zoom_start=12)

        # Add points, size proportional to total connections
        for h3, coord in h3_coords.items():
            total_count = conn_counts.get(h3, 0)
            folium.CircleMarker(
                location=coord[::-1],  # Folium expects [lat, lon]
                radius=3 + total_count**0.3,  # scale size (adjust exponent for visualization)
                color="red",
                fill=True,
                fill_opacity=0.6,
                popup=f"H3: {h3}<br>Total connections: {total_count}"
            ).add_to(m)

        m.save("radplus_points_scaled.html")
        return m

    _()
    return


@app.cell
def _(json):
    def _():
        import folium
        from folium.plugins import MeasureControl

        # Load GeoJSON
        with open("radhackathon.daten/rad+potsdam/p-kreuzungen/data.json") as f:
            geojson_data = json.load(f)

        #geojson_data["features"] = geojson_data["features"][:1000]
        # Create map
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)

        m.add_child(MeasureControl(
            primary_length_unit='kilometers',
            secondary_length_unit='meters',
            primary_area_unit='sqmeters'
        ))
        # Add GeoJSON layer
        folium.GeoJson(
            geojson_data,
            name="GeoJSON Data",
                tooltip=folium.GeoJsonTooltip(
                fields=["route_count", "speed"],   # properties to show
                aliases=["Route Count:", "Average Speed:"],  # labels for display
                localize=True,
                sticky=True
            )
        ).add_to(m)

        # Display in Jupyter or export to HTML
        m.save("map.html")
        return m


    _()
    return


@app.cell
def _(json):
    import folium
    from folium.plugins import MeasureControl
    from folium.plugins import MarkerCluster

    # Load GeoJSON
    with open("radhackathon.daten/rad+potsdam/p-straßensegmente/data.json") as f:
        geojson_data = json.load(f)

    #geojson_data["features"] = geojson_data["features"][:10000]
    # Create map
    map_center = [52.398, 13.065]
    m = folium.Map(location=map_center, zoom_start=12)

    m.add_child(MeasureControl(
        primary_length_unit='kilometers',
        secondary_length_unit='meters',
        primary_area_unit='sqmeters'
    ))
    # Add GeoJSON layer
    folium.GeoJson(
        geojson_data,
        name="GeoJSON Data",
            tooltip=folium.GeoJsonTooltip(
            fields=["route_count", "speed"],   # properties to show
            aliases=["Route Count:", "Average Speed:"],  # labels for display
            localize=True,
            sticky=True
        )
    ).add_to(m)

    # Display in Jupyter or export to HTML
    m.save("map.html")
    m
    return MarkerCluster, folium


@app.cell
def _():
    def _():
        import json
        import folium
        from folium.plugins import MeasureControl

        # Load GeoJSON
        with open("radhackathon.daten/rad+potsdam/p-straßensegmente/data.json") as f:
            geojson_data = json.load(f)

        # Initialize map (centered on Potsdam)
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)

        # Add measure control
        m.add_child(MeasureControl(
            primary_length_unit='kilometers',
            secondary_length_unit='meters',
            primary_area_unit='sqmeters'
        ))

        # Define style function
        def style_function(feature):
            route_count = feature["properties"].get("route_count", 0)
            speed = feature["properties"].get("speed", 0)

            # Scale line thickness (adjust divisor to tune)
            weight = max(1, route_count / 10)

            # Map speed to color
            if speed < 10:
                color = "red"
            elif speed < 20:
                color = "orange"
            elif speed < 30:
                color = "yellow"
            elif speed < 40:
                color = "green"
            else:
                color = "blue"

            return {
                "color": color,
                "weight": weight * 0.05,
                "opacity": 0.8
            }

        # Add GeoJSON layer
        folium.GeoJson(
            geojson_data,
            name="Bike Segments",
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["route_count", "speed"],
                aliases=["Route Count:", "Average Speed:"],
                localize=True,
                sticky=True
            )
        ).add_to(m)

        # Save to file
        m.save("map_streets_usage_speed.html")
        return m

    _()

    return


@app.cell
def _():
    def _():
        import json, folium
        from branca.colormap import LinearColormap
        import numpy as np
        from folium.plugins import MeasureControl

        # Load GeoJSON
        with open("radhackathon.daten/rad+potsdam/p-straßensegmente/data.json") as f:
            geojson_data = json.load(f)

        # Center map around Potsdam
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)

        # Add measure tool
        m.add_child(MeasureControl(
            primary_length_unit='kilometers',
            secondary_length_unit='meters',
            primary_area_unit='sqmeters'
        ))

        # Extract min/max values for scaling
        route_counts = [feat["properties"].get("route_count", 0) for feat in geojson_data["features"]]
        speeds = [feat["properties"].get("speed", 0) for feat in geojson_data["features"]]

        min_count, max_count = min(route_counts), max(route_counts)
        min_speed, max_speed = min(speeds), max(speeds)

        # 🔥 Continuous blue→red color scale for speed
        colormap = LinearColormap(
            #colors=["blue", "cyan", "yellow", "orange", "red"],
            colors=["#440154", "#3B528B", "#21918C", "#5DC863", "#FDE725"],  # vivid viridis-like
            vmin=min_speed,
            vmax=max_speed
        )
        colormap.caption = "Average Speed (km/h)"
        colormap.add_to(m)

        # Function for non-linear thickness scaling (logarithmic)
        def scaled_thickness(value):
            if value <= 0:
                return 1
            return 1 + 8 * np.log10(value - min_count + 1) / np.log10(max_count - min_count + 2)

        # Draw features manually for better control
        for feat in geojson_data["features"]:
            props = feat["properties"]
            geom = feat["geometry"]

            count = props.get("route_count", 0)
            speed = props.get("speed", 0)
            weight = scaled_thickness(count)
            color = colormap(speed)

            if geom["type"] == "LineString":
                folium.PolyLine(
                    locations=[[lat, lon] for lon, lat in geom["coordinates"]],
                    color=color,
                    weight=weight,
                    opacity=0.8,
                    tooltip=folium.Tooltip(
                        f"Route Count: {count}<br>Speed: {speed:.1f} km/h"
                    )
                ).add_to(m)

        # Save or display
        m.save("map_scaled_colored.html")
        return m


    _()

    return


@app.cell
def _():
    import geopandas as gpd

    gdf = gpd.read_file("radhackathon.daten/Radplus/b-city-straßensegmente/data.json")
    sampled_gdf = gdf.sample(n=500, random_state=42)
    sampled_gdf.to_file("sampled_data.json", driver="GeoJSON")
    return


@app.cell
def _():
    # Load your CSV file
    import polars as pl
    import pandas as pd

    #df = pl.read_csv("radhackathon.daten/Unfallatlas/Unfallorte2023_EPSG25832_CSV/csv/Unfallorte2023_LinRef.csv", separator=";")

    df = pd.read_csv("radhackathon.daten/Unfallatlas/Unfallorte2023_EPSG25832_CSV/csv/Unfallorte2023_LinRef.csv", sep=";")
    print(len(df))

    df2 = pd.read_csv("radhackathon.daten/Unfallatlas/Unfallorte2024_EPSG25832_CSV/csv/Unfallorte2024_LinRef.csv", sep=";")
    print(len(df2))



    df = pd.concat([df, df2])

    print(len(df))


    df2
    return df, pd


@app.cell
def _(pd):
    # Load Excel with multiple sheets
    xls = pd.ExcelFile("radhackathon.daten/Verkehrsbereich/Brücken/Brücken2025/hbq5-25.xlsx")

    # Read the sheet you want
    df_source = pd.read_excel(xls, sheet_name="QUERSCHNITT")

    # Copy a specific row (e.g., row index 2)
    row_to_copy = df_source.iloc[23:24]  # slice to keep as DataFrame
    row_to_copy
    return


@app.cell
def _(filtered_df):
    filtered_df.columns
    return


@app.cell
def _(df):
    # Filter for Potsdam (you need the correct UGEMEINDE code for Potsdam)
    # According to the official codes, Potsdam municipality code is likely '12052000' 
    potsdam_code = '12052000'

    df["XGCSWGS84"] = df["XGCSWGS84"].astype("str").str.replace(",", ".").astype(float)
    df["YGCSWGS84"] = df["YGCSWGS84"].astype("str").str.replace(",", ".").astype(float)

    # "03901", "121305"
    # Apply filter for Potsdam and bicycle accidents
    filtered_df = df[
        (df["ULAND"] == 12) &
        (df["UGEMEINDE"] == 0) &
        (df["UKREIS"] == 54) &
        (df['IstRad'] == 1)
    ]


    # Optionally, save the filtered results to a new CSV
    filtered_df.to_csv("potsdam_bicycle_accidents.csv")

    # Display the filtered data
    filtered_df
    return (filtered_df,)


@app.cell
def _(MarkerCluster, filtered_df, folium):
    from folium import Icon

    def display_accidents_with_fatal_icon(filtered_df):
        # Lookup dictionaries for mapped columns
        UKATEGORIE_MAP = {
            1: "Accident with persons killed",
            2: "Accident with seriously injured",
            3: "Accident with slightly injured"
        }

        UART_MAP = {
            1: "Collision with another vehicle which starts, stops or is stationary",
            2: "Collision with another vehicle moving ahead or waiting",
            3: "Collision with another vehicle moving laterally in the same direction",
            4: "Collision with another oncoming vehicle",
            5: "Collision with another vehicle which turns into or crosses a road",
            6: "Collision between vehicle and pedestrian",
            7: "Collision with an obstacle in the carriageway",
            8: "Leaving the carriageway to the right",
            9: "Leaving the carriageway to the left",
            0: "Accident of another kind"
        }

        UTYP1_MAP = {
            1: "Driving accident",
            2: "Accident caused by turning off the road",
            3: "Accident caused by turning into a road or by crossing it",
            4: "Accident caused by crossing the road",
            5: "Accident involving stationary",
            6: "Accident between vehicles moving along in carriageway",
            7: "Other accident"
        }

        USTRZUSTAND_MAP = {
            0: "dry",
            1: "wet/damp/slippery",
            2: "slippery (winter)"
        }

        columns_to_show = [
            'OID_', 'UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG', 
            'UKATEGORIE', 'UART', 'UTYP1', 'ULICHTVERH', 
            'IstStrassenzustand', 'IstRad', 'IstPKW', 'IstFuss',
            'IstKrad', 'IstGkfz', 'IstSonstige'
        ]

        # Initialize map
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered_df.iterrows():
            # Build popup HTML
            popup_html = "<b>Accident Details:</b><br>"
            for col in columns_to_show:
                if col == "UKATEGORIE":
                    value = UKATEGORIE_MAP.get(int(row[col]), row[col])
                elif col == "UART":
                    value = UART_MAP.get(int(row[col]), row[col])
                elif col == "UTYP1":
                    value = UTYP1_MAP.get(int(row[col]), row[col])
                elif col == "IstStrassenzustand":
                    value = USTRZUSTAND_MAP.get(int(row[col]), row[col])
                else:
                    value = row[col]
                popup_html += f"{col}: {value}<br>"

            # Use death icon for fatal accidents, circle marker otherwise
            if int(row["UKATEGORIE"]) == 1:
                folium.Marker(
                    location=[row["YGCSWGS84"], row["XGCSWGS84"]],
                    icon=Icon(icon="skull-crossbones", prefix="fa", color="black"),
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)
            else:
                # Non-fatal accident: adjust color and size
                if int(row["UKATEGORIE"]) == 2:
                    color = "red"      # major accidents
                    radius = 8         # slightly larger
                else:
                    color = "yellow"   # slight accidents
                    radius = 8         # smaller

            
                folium.CircleMarker(
                    location=[row["YGCSWGS84"], row["XGCSWGS84"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(marker_cluster)

        # 1️⃣ Base layer: Stadia Alidade Smooth
        folium.TileLayer(
            tiles='https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> '
                 '&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> '
                 '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="Stadia Alidade Smooth",
            min_zoom=0,
            max_zoom=20,
            control=True
        ).add_to(m)  # Add as base layer

        # 2️⃣ Overlay layer: OpenRailwayMap
        folium.TileLayer(
            tiles='https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
            attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | '
                 'Map style: &copy; <a href="https://www.OpenRailwayMap.org">OpenRailwayMap</a> (CC-BY-SA)',
            name="OpenRailwayMap",
            subdomains=["a", "b", "c"],
            max_zoom=19,
            overlay=True,   # Important: this is a toggleable overlay
            control=True
        ).add_to(m)
    
        # Optional: add layer control
        folium.LayerControl().add_to(m)
        # Save map
        m.save("potsdam_bicycle_accidents_fatal_icon_map.html")

        return m

    # Call the function
    display_accidents_with_fatal_icon(filtered_df)
    return


@app.cell
def _(filtered_df):
    filtered_df.loc[filtered_df['UKATEGORIE'] == 1]
    return


@app.cell
def _(MarkerCluster, filtered_df, folium):
    def _():

        # Lookup dictionaries for mapped columns
        UKATEGORIE_MAP = {
            1: "Accident with persons killed",
            2: "Accident with seriously injured",
            3: "Accident with slightly injured"
        }

        UART_MAP = {
            1: "Collision with another vehicle which starts, stops or is stationary",
            2: "Collision with another vehicle moving ahead or waiting",
            3: "Collision with another vehicle moving laterally in the same direction",
            4: "Collision with another oncoming vehicle",
            5: "Collision with another vehicle which turns into or crosses a road",
            6: "Collision between vehicle and pedestrian",
            7: "Collision with an obstacle in the carriageway",
            8: "Leaving the carriageway to the right",
            9: "Leaving the carriageway to the left",
            0: "Accident of another kind"
        }

        UTYP1_MAP = {
            1: "Driving accident",
            2: "Accident caused by turning off the road",
            3: "Accident caused by turning into a road or by crossing it",
            4: "Accident caused by crossing the road",
            5: "Accident involving stationary",
            6: "Accident between vehicles moving along in carriageway",
            7: "Other accident"
        }

        USTRZUSTAND_MAP = {
            0: "dry",
            1: "wet/damp/slippery",
            2: "slippery (winter)"
        }

        # Columns to show
        columns_to_show = [
            'OID_', 'UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG', 
            'UKATEGORIE', 'UART', 'UTYP1', 'ULICHTVERH', 
            'IstStrassenzustand', 'IstRad', 'IstPKW', 'IstFuss',
                'IstKrad', 'IstGkfz', 'IstSonstige'
            ]

        # Initialize map
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)#, tiles=None)
        marker_cluster = MarkerCluster().add_to(m)

        # Add points
        for _, row in filtered_df.iterrows():
            popup_html = "<b>Accident Details:</b><br>"
            for col in columns_to_show:
                # Map columns if applicable
                if col == "UKATEGORIE":
                    value = UKATEGORIE_MAP.get(int(row[col]), row[col])
                elif col == "UART":
                    value = UART_MAP.get(int(row[col]), row[col])
                elif col == "UTYP1":
                    value = UTYP1_MAP.get(int(row[col]), row[col])
                elif col == "IstStrassenzustand":
                    value = USTRZUSTAND_MAP.get(int(row[col]), row[col])
                else:
                    value = row[col]

                popup_html += f"{col}: {value}<br>"

            folium.CircleMarker(
                location=[row["YGCSWGS84"], row["XGCSWGS84"]],
                radius=4,
                color='red',
                fill=True,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)

        # 1️⃣ Base layer: Stadia Alidade Smooth
        folium.TileLayer(
            tiles='https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> '
                 '&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> '
                 '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="Stadia Alidade Smooth",
            min_zoom=0,
            max_zoom=20,
            control=True
        ).add_to(m)  # Add as base layer

        # 2️⃣ Overlay layer: OpenRailwayMap
        folium.TileLayer(
            tiles='https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
            attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | '
                 'Map style: &copy; <a href="https://www.OpenRailwayMap.org">OpenRailwayMap</a> (CC-BY-SA)',
            name="OpenRailwayMap",
            subdomains=["a", "b", "c"],
            max_zoom=19,
            overlay=True,   # Important: this is a toggleable overlay
            control=True
        ).add_to(m)
    
        # Optional: add layer control
        folium.LayerControl().add_to(m)
        # Save map as HTML
        m.save("potsdam_bicycle_accidents_clustered_map.html")

        # Display map in Jupyter Notebook (if using notebook)
        return m


    _()
    return


@app.cell
def _(filtered_df):
    filtered_df["UJAHR"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
