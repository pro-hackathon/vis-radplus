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

        geojson_data["features"] = geojson_data["features"]
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

    geojson_data["features"] = geojson_data["features"][:10000]
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
    return (df,)


@app.cell
def _():
    import pandas as pd
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
        (df['IstRad'] == 1)
    ]


    # Optionally, save the filtered results to a new CSV
    filtered_df.to_csv("potsdam_bicycle_accidents.csv")

    # Display the filtered data
    filtered_df
    return (filtered_df,)


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
        m = folium.Map(location=map_center, zoom_start=12)
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
