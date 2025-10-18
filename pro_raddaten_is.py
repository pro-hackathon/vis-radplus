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
    from pathlib import Path
    return (
        DivIcon,
        LinearColormap,
        MarkerCluster,
        Path,
        folium,
        json,
        mo,
        np,
        pd,
    )


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
    def get_marker_svg(size):
        return f"""

    <svg
       width="{str(size)}mm"
       height="{str(size)}mm"
       viewBox="0 0 210 297"
       version="1.1"
       id="svg1"
       xml:space="preserve"
       inkscape:version="1.4.2 (ebf0e940, 2025-05-08)"
       sodipodi:docname="marker.svg"
       xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
       xmlns="http://www.w3.org/2000/svg"
       xmlns:svg="http://www.w3.org/2000/svg"><defs
         id="defs1" /><g
         inkscape:label="Layer 1"
         inkscape:groupmode="layer"
         id="layer1"><path
           id="path1"
           style="fill:#333333;stroke-width:1.72024"
           d="M 105.45713 0.84749349 C 98.368597 0.8355104 91.247348 1.3691799 83.99384 2.4437785 C 40.623829 8.8690054 12.242628 33.076287 3.175 71.378072 C -1.0782704 89.343992 -0.19127461 117.59838 5.2162191 136.38599 C 14.172112 167.50209 37.698108 208.62431 75.738013 259.65754 C 83.088967 269.51938 92.476288 281.64951 96.721228 286.77009 C 97.986066 288.29583 100.35337 291.18854 101.98137 293.19864 C 104.17663 295.90917 104.94292 296.64082 105.45713 296.41808 L 105.45713 159.91789 C 103.96795 159.98138 102.93226 159.91273 101.07032 159.73547 C 71.840923 156.95379 50.495433 133.95328 50.472371 105.21528 C 50.465671 96.819551 52.451141 87.888491 55.805379 81.225016 C 59.420094 74.044109 66.147656 65.941582 72.349072 61.302201 C 78.210237 56.917352 86.68867 53.1168 94.239726 51.487813 C 97.040022 50.883707 101.26289 50.603821 105.45713 50.635669 L 105.45713 0.84749349 z " /><path
           id="path1-9"
           style="fill:#4d4d4d;stroke-width:1.72024"
           d="M 105.45713 0.84749349 L 105.45713 50.635669 C 105.47465 50.635802 105.4918 50.635525 105.50932 50.635669 L 105.50932 50.636186 C 109.72083 50.670776 113.89798 51.019592 116.5686 51.671265 C 134.02168 55.930071 146.95677 66.154477 154.38282 81.560396 C 158.54458 90.194273 160.11255 98.007565 159.67656 107.94483 C 158.48666 135.06652 137.32287 156.96431 109.7778 159.57372 C 107.80221 159.76085 106.51524 159.87277 105.45713 159.91789 L 105.45713 296.41808 C 105.50211 296.3986 105.54492 296.37191 105.58684 296.3385 C 105.94213 296.05525 110.55997 290.39854 115.84874 283.76821 C 157.03459 232.13505 187.30969 184.28859 200.47458 150.02443 C 204.87108 138.58167 206.87323 130.57797 208.841 116.58823 C 209.99844 108.36005 210.0159 88.855161 208.87304 81.306148 C 207.94562 75.180151 205.82033 66.185093 204.1493 61.316671 C 202.54127 56.631828 198.13099 47.495845 195.54104 43.484684 C 183.05823 24.151884 164.08817 11.116985 139.74031 5.1423218 C 130.31292 2.828961 121.09913 1.4545036 111.88103 1.0082072 C 109.74073 0.90458455 107.60066 0.85111711 105.45713 0.84749349 z " /></g></svg>

    """

    scale = 0.2

    def add_knoten_data(knoten_layer):
        # Load CSV
        df = pd.read_csv("merged_knoten_data.csv")
        for _, row in df.iterrows():
            lat, lon = map(float, row["Geo Point"].split(","))

            # Non-linear scaling for size
            size = scale * (row["Summe"] ** 0.5)
            print(size)

            popup_html = f"<b>{row['Name']}</b><br>Year: {row['Jahr']}<br>Count: {row['Summe']}"

            # Use Unicode pin instead of FontAwesome
            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(size, size),
                    icon_anchor=(size//2, size),  # anchor at bottom
                    html=f"<div>{get_marker_svg(size)}</div>"
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
            size = scale * (row["sum_value"] ** 0.5)  # adjust scaling if needed

            popup_html = f"<b>{row['bridge_name'].title()}</b><br>Year: {row['year_full']}<br>Sum: {row['sum_value']}"

            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(size, size),
                    icon_anchor=(size//2, size),
                    html=f"<div>{get_marker_svg(size)}</div>"

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
    def summarize_speed_histogram(hist_counts):
        """
        Given a histogram of counts per speed bin (0..29 km/h),
        return min, 25%, median, 75%, max, and mean speeds.
        """
        print("Histogram length:", len(hist_counts))  # should be 30

        if not hist_counts or sum(hist_counts) == 0:
            return pd.Series({
                "speed_min": np.nan,
                "speed_25": np.nan,
                "speed_median": np.nan,
                "speed_mean": np.nan,
                "speed_75": np.nan,
                "speed_max": np.nan
            })
    
        # Create the speed bins array: 0, 1, 2, ..., len(hist_counts)-1
        bins = np.arange(len(hist_counts))
        counts = np.array(hist_counts)
    
        # Compute cumulative distribution
        cum_counts = np.cumsum(counts)
        total = cum_counts[-1]
    
        # Helper function to find speed at a given percentile
        def percentile(p):
            idx = np.searchsorted(cum_counts, p * total / 100)
            return bins[min(idx, len(bins)-1)]
    
        # Weighted mean
        mean_speed = np.sum(bins * counts) / total
    
        return pd.Series({
            "speed_min": bins[np.argmax(counts > 0)],
            "speed_25": percentile(25),
            "speed_median": percentile(50),
            "speed_mean": mean_speed,
            "speed_75": percentile(75),
            "speed_max": bins[np.max(np.where(counts > 0))]
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
        df = pd.concat([df, df["speeds"].apply(summarize_speed_histogram)], axis=1)
        # Display summary statistics
        print(df.describe())

        print(df.loc[df["route_count"] >= 50].describe())
        return df

    radplus_data = load_radplus()
    radplus_data
    return (radplus_data,)


@app.cell
def _(LinearColormap, folium):
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
        if not speed_threshold:
            colormap.caption = f"Average Speed ({speed_column}) [km/h]"
            colormap.add_to(m)

        def scaled_thickness(value, min_count=200, max_count=40000, min_width=1, max_width=25):
            """
            Scale line thickness linearly from min_width to max_width
            based on value between min_count and max_count.
            """
            if value <= min_count:
                return min_width
            elif value >= max_count:
                return max_width
            else:
                # linear interpolation
                return min_width + (value - min_count) / (max_count - min_count) * (max_width - min_width)

        # 🛣️ Draw each segment
        for _, row in radplus_data.iterrows():
            coords = row["coordinates"]
            if not isinstance(coords, list) or len(coords) == 0:
                continue

            speed_value = row.get(speed_column, 0)
            color = colormap(speed_value)
            if not speed_threshold:
                weight = scaled_thickness(row["route_count"], min_count, max_count)
            else:
                weight = scaled_thickness(row["route_count"], min_count, max_count, min_width=2, max_width=10)
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
                color=color if not speed_threshold else "blue",
                weight=weight,
                opacity=0.8,
                tooltip=folium.Tooltip(tooltip_text)
            ).add_to(radplus_layer)


    return (add_radplus_data,)


@app.cell
def _(Path, folium, json):
    def add_geojson_layers(
        m,
        layers_info,
        weight=5,
        opacity=0.7
    ):

        for layer in layers_info:
            file_path = Path(layer["file"])
            layer_name = layer.get("name", file_path.stem)

            # FeatureGroup, damit Layer einzeln toggelbar sind
            fg = folium.FeatureGroup(name=layer_name, show=False).add_to(m)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)
            except FileNotFoundError:
                print(f"⚠️ Datei nicht gefunden: {file_path}")
                continue

            for feat in geojson_data.get("features", []):
                geom = feat.get("geometry")
                props = feat.get("properties", {})

                if geom and geom["type"] == "LineString":
                    folium.PolyLine(
                        locations=[[lat, lon] for lon, lat in geom["coordinates"]],
                        color=layer["color"],
                        weight=weight,
                        opacity=opacity,
                        tooltip=folium.Tooltip(
                            f"Surface: {props.get('surface', 'n/a')}<br>"
                            f"Segregated: {props.get('segregated', 'n/a')}<br>"
                            f"Traffic sign: {props.get('traffic_sign', 'n/a')}"
                        )
                    ).add_to(fg)

        return m
    return (add_geojson_layers,)


@app.cell
def _(folium, json):
    def add_unlit_bike_paths(m, geojson_file="unbeleuchteteRadWege.geojson", layer_name="Unlit Bike Paths"):
        """
        Add unlit bicycle paths from a GeoJSON file to a Folium map as a separate layer.
        """
        # Create a feature group for toggling
        unlit_layer = folium.FeatureGroup(name=f"{layer_name} 🛣️").add_to(m)

        # Load GeoJSON
        with open(geojson_file, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # Add each LineString feature
        for feat in geojson_data["features"]:
            geom = feat["geometry"]
            props = feat.get("properties", {})
        
            if geom["type"] == "LineString":
                folium.PolyLine(
                    locations=[[lat, lon] for lon, lat in geom["coordinates"]],
                    color="darkblue",          # pick a color that stands out
                    weight=5,                  # thickness of the line
                    opacity=0.7,
                    tooltip=folium.Tooltip(
                        f"Surface: {props.get('surface', 'n/a')}<br>"
                        f"Segregated: {props.get('segregated', 'n/a')}<br>"
                        f"Traffic sign: {props.get('traffic_sign', 'n/a')}"
                    )
                ).add_to(unlit_layer)
        return unlit_layer
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #  🥴 🤕 💀 <br/>Wie gefährlich ist Potsdam für Radfahrende?!

    Im Zeitraum **2023–2024** wurden **745 Unfälle** mit Fahrradbeteiligung registriert, davon ereigneten sich **687 leichte**, **56 schwere** und **2 tödliche Unfälle**. 

    Dies entspricht einem Durchschnitt von etwa **ein Unfall pro Tag** – mit tragischerweise **einem tödlichen Unfall pro Jahr**.
    """
    )
    return


@app.cell
def _(mo):
    mo.image(src="Bild3.png")
    return


@app.cell
def _(
    MarkerCluster,
    add_accident_data,
    add_geojson_layers,
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
            show=False,
            overlay=True,   # Important: this is a toggleable overlay
            control=True
        ).add_to(m)

        ####### Accident data
        # Feature group for deaths (always visible)
        deaths_layer = folium.FeatureGroup(name="Deaths", show=True)

        # MarkerCluster for non-fatal accidents
        accidents_cluster = MarkerCluster(name="Accidents", show=False).add_to(m)

        for _, row in filtered_df.iterrows():
            add_accident_data(row, deaths_layer, accidents_cluster)

        # Add deaths layer separately
        deaths_layer.add_to(m)

        ########## Add (manual) Verkehszählung: Knoten/Brücken Data
        # FeatureGroup for toggle
        knoten_layer = folium.FeatureGroup(name="Traffic Counts", show=False).add_to(m)
        add_knoten_data(knoten_layer)

        ####### Add DB RadPlus data
        # Options: "speed", "speed_min", "speed_25", "speed_median", "speed_75", "speed_max"
        # 🎨 Create feature layer
        radplus_layer_mean = folium.FeatureGroup(name=f"DB Rad+ (Mean, >100)", show=False).add_to(m)
        add_radplus_data(m, radplus_layer_mean, radplus_data, "speed", min_count_threshold=100)

        radplus_layer_median = folium.FeatureGroup(name=f"DB Rad+ (Med<10, >100)", show=False).add_to(m)
        add_radplus_data(m, radplus_layer_median, radplus_data, "speed_median", min_count_threshold=100, speed_threshold=10)

        #unlit_layer = add_unlit_bike_paths(m, "unbeleuchteteRadWege.geojson", "Unlit Bike Paths")

        layers = [
            {"file": "unbeleuchteteRadWege.geojson", "name": "Blindfahrt: Unbeleuchtete Radwege","color": "#9B5DE5"},
            {"file": "Laub.geojson", "name": "Blätterchaos: Laub auf Radwegen", "color": "#C65CCD"},
            {"file": "SchmaleRadwege.geojson", "name": "Engpass: Schmale Radwege (<1,5m)", "color": "#F15BB5"},
            {"file": "ParkstreifenRechts.geojson", "name": "Dooring-Alarm: parkende Autos","color":"#3DA3E8"},
            {"file": "SchlechterUntergrund.geojson", "name": "Holperpiste: Mangelhafter Straßenzustand", "color":"#99E6FF"},
            {"file": "AbbiegenKreuzungen.geojson", "name": "Kreuzungsdrama: potenzielle Abbiegekonflikte", "color":"#00D8E7"},
            {"file": "Straßen-schneller-als-30.geojson", "name": "Vollgaszone: Autos fahren 50 km/h","color":"#00F5D4"},
    
        ]
    
        add_geojson_layers(m, layers)


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
