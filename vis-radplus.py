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
        with open("radhackathon.daten/Radplus/b-city-kreuzungen/data.json") as f:
            geojson_data = json.load(f)

        geojson_data["features"] = geojson_data["features"][:1000]
        # Create map
        m = folium.Map(location=[20, 0], zoom_start=2)

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

    # Load GeoJSON
    with open("radhackathon.daten/Radplus/b-city-straßensegmente/data.json") as f:
        geojson_data = json.load(f)

    geojson_data["features"] = geojson_data["features"][:1000]
    # Create map
    m = folium.Map(location=[20, 0], zoom_start=2)

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

    return


@app.cell
def _():
    import geopandas as gpd

    gdf = gpd.read_file("radhackathon.daten/Radplus/b-city-straßensegmente/data.json")
    sampled_gdf = gdf.sample(n=500, random_state=42)
    sampled_gdf.to_file("sampled_data.json", driver="GeoJSON")
    return


if __name__ == "__main__":
    app.run()
