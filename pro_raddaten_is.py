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
    # filtered_df
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
       viewBox="0 0 297 297"
       version="1.1"
       id="svg1"
       xml:space="preserve"
       inkscape:version="1.4.2 (ebf0e940, 2025-05-08)"
       sodipodi:docname="marker.svg"
       xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
       xmlns="http://www.w3.org/2000/svg"
       xmlns:svg="http://www.w3.org/2000/svg"><defs
         id="defs1" />  <g
         inkscape:label="Layer 1"
         inkscape:groupmode="layer"
         id="layer1">
        <circle
           style="fill:#395e58;stroke-width:0.264583"
           id="path3"
           cx="148.5"
           cy="148.5"
           r="146.84698" />
        <circle
           style="fill:#5e9e94;stroke-width:0.264583"
           id="path4"
           cx="148.5"
           cy="148.5"
           r="94.328796" />
      </g></svg>

    """

    scale = 0.1
    opacity = 0.8

    def add_knoten_data(knoten_layer):
        # Load CSV
        df = pd.read_csv("merged_knoten_data.csv")
        for _, row in df.iterrows():
            lat, lon = map(float, row["Geo Point"].split(","))

            # Non-linear scaling for size
            size = scale * (row["Summe"] ** 0.5)

            popup_html = f"<b>{row['Name']}</b><br>Year: {row['Jahr']}<br>Count: {row['Summe']}"

            # Use Unicode pin instead of FontAwesome
            folium.Marker(
                location=[lat, lon],
                opacity=opacity,
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
                opacity=opacity,
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
    #print(stat_dbplus.describe())
    #stat_dbplus
    return


@app.cell
def _(json, np, pd):
    def summarize_speed_histogram(hist_counts):
        """
        Given a histogram of counts per speed bin (0..29 km/h),
        return min, 25%, median, 75%, max, and mean speeds.
        """

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
        #print(df.describe())

        #print(df.loc[df["route_count"] >= 50].describe())
        return df

    radplus_data = load_radplus()
    #radplus_data
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
def _(mo):
    mo.Html(
    """
    <style>
      /* --- General page styling --- */
      body {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        padding: 50px 20px;        /* space around content */
      }

      /* --- Headline styling --- */
      h1 {
        font-family: 'Courier', Arial, serif;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 0.3em;
      }

      /* --- Emoji section above the headline --- */
      .emojis {
        font-size: 2em;
        margin-bottom: 0.2em;
      }

      /* --- Subheadline / main text block --- */
      .content {
        font-size: 1.2em;
        max-width: 700px;
        margin: 0 auto 2em auto;
        line-height: 1.6;
      }

      /* --- Highlight important numbers or words --- */
      strong {
        font-weight: 700;
      }

      /* --- Legend styling at the bottom --- */
      .legend {
        font-size: 1em;
        margin-top: 0em;
        max-width: fit-content;
        margin-left: auto;
        margin-right: auto;
      }
    </style>
    </head>

    <body>

      <!-- Emoji row -->
      <div class="emojis">🤔🤕💀</div>

      <!-- Main headline -->
      <h1>Wie gefährlich ist Potsdam<br>für Radfahrende?</h1>

      <!-- Main text block -->
      <div class="content">
        Im Zeitraum <strong>2023–2024</strong> wurden <strong>745 Unfälle</strong> mit Fahrradbeteiligung registriert, davon ereigneten sich <strong>687 leichte</strong>, <strong>56 schwere</strong> und <strong>2 tödliche</strong> Unfälle.<br>Dies entspricht einem Durchschnitt von etwa <strong>einem Unfall pro Tag</strong> – mit tragischerweise <strong>einem tödlichen Unfall pro Jahr</strong>.
      </div>

      <!-- Legend at bottom -->
      <div class="legend">
        <strong>Legende</strong><br>
        leicht verletzt 🤕<br>
        schwer verletzt 🤔<br>
        tot 💀
      </div>

    """
    )
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
        m = folium.Map(location=map_center, zoom_start=12, attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | Unfallatlas data: &copy; <a href="https://www.govdata.de/dl-de/by-2-0">Datenlizenz Deutschland – Namensnennung – Version 2.0</a> <a href="https://unfallatlas.statistikportal.de/">Unfallatlas</a> | Traffic Count data: &copy; <a href="https://ckan.urbanedatenplattform-potsdam.de/dataset/verkehrszahlungen">LHP</a> | DB Rad+ data: &copy; DB')

        # 1️⃣ Base layer: Stadia Alidade Smooth
        folium.TileLayer(
            tiles='https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> '
                 '&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> '
                 '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="Stadia Alidade Smooth",
            min_zoom=0,
            max_zoom=20,
            show=False,
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
        radplus_layer_mean = folium.FeatureGroup(name=f"DB Rad+ (Mean, >500)", show=False).add_to(m)
        add_radplus_data(m, radplus_layer_mean, radplus_data, "speed", min_count_threshold=500)

        radplus_layer_median = folium.FeatureGroup(name=f"DB Rad+ (Med<10, >500)", show=False).add_to(m)
        add_radplus_data(m, radplus_layer_median, radplus_data, "speed_median", min_count_threshold=500, speed_threshold=10)

        #unlit_layer = add_unlit_bike_paths(m, "unbeleuchteteRadWege.geojson", "Unlit Bike Paths")

        layers = [
            {"file": "osm_data/unbeleuchteteRadWege.geojson", "name": "Blindfahrt: Unbeleuchtete Radwege","color": "#9B5DE5"},
            {"file": "osm_data/Laub.geojson", "name": "Blätterchaos: Laub auf Radwegen", "color": "#C65CCD"},
            {"file": "osm_data/SchmaleRadwege.geojson", "name": "Engpass: Schmale Radwege (<1,5m)", "color": "#F15BB5"},
            {"file": "osm_data/ParkstreifenRechts.geojson", "name": "Dooring-Alarm: parkende Autos","color":"#3DA3E8"},
            {"file": "osm_data/SchlechterUntergrund.geojson", "name": "Holperpiste: Mangelhafter Straßenzustand", "color":"#99E6FF"},
            {"file": "osm_data/AbbiegenKreuzungen.geojson", "name": "Kreuzungsdrama: potenzielle Abbiegekonflikte", "color":"#00D8E7"},
            {"file": "osm_data/Straßen-schneller-als-30.geojson", "name": "Vollgaszone: Autos fahren 50 km/h","color":"#00F5D4"},

        ]

        add_geojson_layers(m, layers)


        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        m.save("maps/potsdam_bicycle_accidents_fatal_icon_map.html")
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
