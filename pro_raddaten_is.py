import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import glob
    import json
    import folium
    from folium.plugins import MeasureControl
    from folium.plugins import MarkerCluster
    from branca.colormap import LinearColormap
    import numpy as np
    import pandas as pd
    from folium import Icon
    from folium.features import DivIcon
    return DivIcon, LinearColormap, MarkerCluster, folium, json, mo, np, pd


@app.cell
def _(pd):
    # Load Unfalldaten

    df = pd.read_csv("radhackathon.daten/Unfallatlas/Unfallorte2023_EPSG25832_CSV/csv/Unfallorte2023_LinRef.csv", sep=";")
    df2 = pd.read_csv("radhackathon.daten/Unfallatlas/Unfallorte2024_EPSG25832_CSV/csv/Unfallorte2024_LinRef.csv", sep=";")
    df = pd.concat([df, df2])
    # Filter for Potsdam (you need the correct UGEMEINDE code for Potsdam)

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

    columns_to_show = [
        'OID_', 'UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG', 
        'UKATEGORIE', 'UART', 'UTYP1', 'ULICHTVERH', 
        'IstStrassenzustand', 'IstRad', 'IstPKW', 'IstFuss',
        'IstKrad', 'IstGkfz', 'IstSonstige'
    ]
    return (
        UART_MAP,
        UKATEGORIE_MAP,
        USTRZUSTAND_MAP,
        UTYP1_MAP,
        columns_to_show,
    )


@app.cell
def _(
    DivIcon,
    UART_MAP,
    UKATEGORIE_MAP,
    USTRZUSTAND_MAP,
    UTYP1_MAP,
    columns_to_show,
    folium,
):
    def add_accident_data(row, deaths_layer, accidents_cluster):
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



        if int(row["UKATEGORIE"]) == 1:
            # Fatal accident
            emoji = "💀"
            folium.Marker(
                location=[row["YGCSWGS84"], row["XGCSWGS84"]],
                icon=DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html=f"<div style='font-size:24px'>{emoji}</div>"
                ),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(deaths_layer)
        else:
            # Non-fatal accident
            emoji = "🤕" if row["UKATEGORIE"] == 2 else "🥴"
            folium.Marker(
                location=[row["YGCSWGS84"], row["XGCSWGS84"]],
                icon=DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html=f"<div style='font-size:20px'>{emoji}</div>"
                ),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(accidents_cluster)
    return (add_accident_data,)


@app.cell
def _(DivIcon, folium, pd):
    def add_knoten_data(knoten_layer):
        # Load CSV
        df = pd.read_csv("merged_knoten_data.csv")
        for _, row in df.iterrows():
            lat, lon = map(float, row["Geo Point"].split(","))

            # Non-linear scaling for size
            size = 15 + (row["Summe"] ** 0.5) / 2  # adjust 15/base and /2 as needed

            popup_html = f"<b>{row['Name']}</b><br>Year: {row['Jahr']}<br>Count: {row['Summe']}"

            # Use Unicode pin instead of FontAwesome
            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(size, size),
                    icon_anchor=(size//2, size),  # anchor at bottom
                    html=f"""<div style="font-size:{size}px; line-height:1; color:green;">
                                📍
                             </div>"""
                ),
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(knoten_layer)
        # Load the new CSV
        df_bruecken = pd.read_csv("bruecken_sums_yearly.csv")
        # Convert two-digit years to full years
        def full_year(y):
            y = int(y)
            if y < 50:      # assume 00–49 → 2000–2049
                return 2000 + y
            else:           # assume 50–99 → 1950–1999
                return 1900 + y

        df_bruecken["year_full"] = df_bruecken["year"].apply(full_year)

        # Filter only newest year
        df_latest = df_bruecken.loc[df_bruecken.groupby("bridge_name")["year_full"].idxmax()]

        # Add these points to the existing knoten_layer
        for _, row in df_latest.iterrows():
            lat, lon = row["latitude"], row["longitude"]

            # Scale size relative to sum_value (same formula as before)
            size = 15 + (row["sum_value"] ** 0.5) / 2  # adjust scaling if needed

            popup_html = f"<b>{row['bridge_name'].title()}</b><br>Year: {row['year_full']}<br>Sum: {row['sum_value']}"

            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(size, size),
                    icon_anchor=(size//2, size),
                    html=f"""<div style="font-size:{size}px; line-height:1; color:green;">
                                📍
                             </div>"""
                ),
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(knoten_layer)
    return (add_knoten_data,)


@app.cell
def _(json, pd):
    with open("radhackathon.daten/rad+potsdam/p-straßensegmente/data.json") as f:
        geojson_data = json.load(f)
    # Extract min/max values for scaling
    route_counts = [feat["properties"].get("route_count", 0) for feat in geojson_data["features"]]
    speeds = [feat["properties"].get("speed", 0) for feat in geojson_data["features"]]

    stat_dbplus = pd.DataFrame({"counts": route_counts, "speeds": speeds})
    print(stat_dbplus.describe())
    stat_dbplus
    return


@app.cell
def _(json, np, pd):
    # assuming df["speeds"] is a list/array of numeric values for each row
    def summarize_speeds(speeds):
        if not speeds or len(speeds) == 0:
            return pd.Series({
                "speed_min": np.nan,
                "speed_25": np.nan,
                "speed_median": np.nan,
                "speed_75": np.nan,
                "speed_max": np.nan
            })
        speeds = np.array(speeds)
        return pd.Series({
            "speed_min": np.min(speeds),
            "speed_25": np.percentile(speeds, 25),
            "speed_median": np.median(speeds),
            "speed_75": np.percentile(speeds, 75),
            "speed_max": np.max(speeds)
        })



    def load_radplus():
        with open("radhackathon.daten/rad+potsdam/p-straßensegmente/data.json") as f:
            geojson_data = json.load(f)
        # Flatten GeoJSON features into a DataFrame
        df = pd.json_normalize(geojson_data["features"])
    
        # Optionally, rename columns for readability
        df.columns = df.columns.str.replace("properties.", "", regex=False)
        df.columns = df.columns.str.replace("geometry.", "", regex=False)
    
        # Apply the summary function to each row
        df = pd.concat([df, df["speeds"].apply(summarize_speeds)], axis=1)
        # Display summary statistics
        print(df.describe())

        print(df.loc[df["route_count"] >= 50].describe())
        return df

    radplus_data = load_radplus()
    radplus_data
    return (radplus_data,)


@app.cell
def _(LinearColormap, folium, np):
    def add_radplus_data(m, radplus_layer, radplus_data, speed_column="speed_median", min_count_threshold=50, speed_threshold=None):
        """
        Add DB Rad+ road segment data as a Folium layer.
    
        Parameters:
            m (folium.Map): The map object.
            radplus_data (pd.DataFrame): Preprocessed GeoJSON data flattened into a DataFrame.
            speed_column (str): Which speed column to use for color scaling (e.g., "speed", "speed_median").
            min_count_threshold (int): Minimum route_count to display.
        """
    
        # ⚡ Apply threshold at the beginning
        radplus_data = radplus_data[radplus_data["route_count"] >= min_count_threshold].copy()
        if speed_threshold:
            radplus_data = radplus_data.loc[radplus_data[speed_column] < speed_threshold]
    


        # Extract min/max values for color and thickness scaling
        min_count, max_count = radplus_data["route_count"].min(), radplus_data["route_count"].max()
        min_speed, max_speed = radplus_data[speed_column].min(), radplus_data[speed_column].max()

        # 🔥 Continuous blue→red color scale for selected speed metric
        colormap = LinearColormap(
            colors=["blue", "yellow", "red"],
            vmin=min_speed,
            vmax=max_speed
        )
        colormap.caption = f"Average Speed ({speed_column}) [km/h]"
        colormap.add_to(m)

        # ⚖️ Nonlinear scaling for line thickness
        def scaled_thickness(value):
            if value <= 0:
                return 1
            return 1 + 10 * np.log10(value - min_count + 1) / np.log10(max_count - min_count + 2)

        # 🛣️ Draw each segment
        for _, row in radplus_data.iterrows():
            coords = row["coordinates"]
            if not isinstance(coords, list) or len(coords) == 0:
                continue

            speed_value = row.get(speed_column, 0)
            color = colormap(speed_value)
            weight = scaled_thickness(row["route_count"])

            # 📊 Tooltip with all available speed stats
            tooltip_text = (
                f"<b>Route Count:</b> {row['route_count']}<br>"
                f"<b>Speeds (km/h):</b><br>"
                f"• Min: {row.get('speed_min', 0):.1f}<br>"
                f"• 25%: {row.get('speed_25', 0):.1f}<br>"
                f"• Median: {row.get('speed_median', 0):.1f}<br>"
                f"• 75%: {row.get('speed_75', 0):.1f}<br>"
                f"• Max: {row.get('speed_max', 0):.1f}<br>"
            )

            folium.PolyLine(
                locations=[[lat, lon] for lon, lat in coords],
                color=color,
                weight=weight,
                opacity=0.8,
                tooltip=folium.Tooltip(tooltip_text)
            ).add_to(radplus_layer)


    return (add_radplus_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Raddaten Hackathon

    🥴🥴🥴🥴🥴🥴🥴🥴🥴
    """
    )
    return


@app.cell
def _(
    MarkerCluster,
    add_accident_data,
    add_knoten_data,
    add_radplus_data,
    filtered_df,
    folium,
    radplus_data,
):
    def display_accidents_with_fatal_icon(filtered_df):
        # Initialize map
        map_center = [52.398, 13.065]
        m = folium.Map(location=map_center, zoom_start=12)

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

        ####### Accident data
        # Feature group for deaths (always visible)
        deaths_layer = folium.FeatureGroup(name="Deaths", show=True)

        # MarkerCluster for non-fatal accidents
        accidents_cluster = MarkerCluster(name="Accidents").add_to(m)

        for _, row in filtered_df.iterrows():
            add_accident_data(row, deaths_layer, accidents_cluster)

        # Add deaths layer separately
        deaths_layer.add_to(m)

        ########## Add (manual) Verkehszählung: Knoten/Brücken Data
        # FeatureGroup for toggle
        knoten_layer = folium.FeatureGroup(name="Traffic Counts").add_to(m)
        add_knoten_data(knoten_layer)

        ####### Add DB RadPlus data
        # Options: "speed", "speed_min", "speed_25", "speed_median", "speed_75", "speed_max"
        # 🎨 Create feature layer
        #radplus_layer_mean = folium.FeatureGroup(name=f"DB Rad+ (Mean, >200)").add_to(m)
        #add_radplus_data(m, radplus_layer_mean, radplus_data, "speed", min_count_threshold=200)

        radplus_layer_median = folium.FeatureGroup(name=f"DB Rad+ (Med<10, >200)").add_to(m)
        add_radplus_data(m, radplus_layer_median, radplus_data, "speed_median", min_count_threshold=200, speed_threshold=10)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        m.save("potsdam_bicycle_accidents_fatal_icon_map.html")
        print("Done")

        return m

    # Call the function
    display_accidents_with_fatal_icon(filtered_df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
