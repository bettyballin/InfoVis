import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import geoplotlib

def vis():

    df = pd.read_csv("route1.csv",delimiter=";")
    df["Latitude"] = df["Latitude"].apply(lambda x: float(x.replace(",",".")))
    df["Longitude"] = df["Longitude"].apply(lambda x: float(x.replace(",",".")))
    df["lat"] = df["Latitude"].shift(-1)
    df["lon"] = df["Longitude"].shift(-1)
    df.dropna(subset = ["lat","lon"], inplace=True)
    #df.at[0,"lat"] = df.at[1,"Latitude"]
    #df.at[0,"lon"] = df.at[1,"Longitude"]
    print(df)
    geoplotlib.graph(df, src_lat='Latitude', src_lon='Longitude', dest_lat='lat', dest_lon='lon', color='hot_r', alpha=50, linewidth=20)
    geoplotlib.show()

def visualize():
    
    df = pd.read_csv("route1.csv",delimiter=";")
    df["Latitude"] = df["Latitude"].apply(lambda x: float(x.replace(",",".")))
    df["Longitude"] = df["Longitude"].apply(lambda x: float(x.replace(",",".")))
    print(df.head(10))
    fig = go.Figure(data=go.Scattergeo(
        lat = df["Latitude"],
        lon = df["Longitude"],
        mode = 'lines',
        line = dict(width = 2, color = 'blue'),
    ))

    fig.update_layout(
        title_text = 'Route des Weihnachtsmanns',
        showlegend = False,
        mapbox = {
            'center': {'lon': 5, 'lat': 40},
            'style': "stamen-terrain",
            'center': {'lon': 20, 'lat': 60},
            'zoom': 3},
        geo = dict(
            resolution = 50,
            showland = True,
            showlakes = True,
            landcolor = 'rgb(204, 204, 204)',
            countrycolor = 'rgb(204, 204, 204)',
            lakecolor = 'rgb(255, 255, 255)',
            projection_type = "equirectangular",
            coastlinewidth = 2,
            lataxis = dict(
                range = [40, 60],
                showgrid = True,
                dtick = 10
            ),
            lonaxis = dict(
                range = [5, 20],
                showgrid = True,
                dtick = 20
            ),
        )
    )
    fig.show()

def vis2():
    locations = pd.read_csv("route2.csv", delimiter=";")
    locations["Latitude"] = locations["Latitude"].apply(lambda x: float(x.replace(",", ".")))
    locations["Longitude"] = locations["Longitude"].apply(lambda x: float(x.replace(",", ".")))

    scale = 5000
    fig = go.Figure()
    
    last_lon = locations['Longitude'][0]
    last_lat = locations['Latitude'][0]
    for i in range(len(locations)-1):
        if abs(last_lon-locations['Longitude'][i+1]) > 0.01 and abs(last_lat-locations['Latitude'][i+1]) > 0.01:
            fig.add_trace(go.Scattergeo(
                lon=[last_lon],
                lat=[last_lat],
                text=[" x"],
                textfont={"color": 'green',
                        "family": 'Droid Serif',
                        "size": 10},
                textposition="top center",
                name="Candidate Facility",
                mode="markers",
                marker=dict(
                    size=1,
                    color="red",
                    line_color='blue',
                    symbol = 'cross',
                    sizemode='area')))

            fig.add_trace(go.Scattergeo(
                lat=[last_lat, locations['Latitude'][i+1]],
                lon=[last_lon, locations['Longitude'][i+1]],
                mode='lines',
                line=dict(width=1, color='black'),
            ))

            # Workaround to get the arrow at the end of an edge AB

            l = 0.1  # the arrow length
            widh = 0.035  # 2*widh is the width of the arrow base as triangle

            A = np.array([last_lon, last_lat])
            B = np.array([locations['Longitude'][i+1], locations['Latitude'][i+1]])
            v = B - A
            w = v / np.linalg.norm(v)
            u = np.array([-v[1], v[0]])  # u orthogonal on  w

            P = B - l * w
            S = P - widh * u
            T = P + widh * u

            fig.add_trace(go.Scattergeo(lon=[S[0], T[0], B[0], S[0]],
                                        lat=[S[1], T[1], B[1], S[1]],
                                        mode='lines',
                                        fill='toself',
                                        fillcolor='black',
                                        line_color='black'))
    
            last_lon = locations['Longitude'][i+1]
            last_lat = locations['Latitude'][i+1]
    fig.update_layout(
        showlegend=False,
        geo=dict(
            resolution = 50,
            showland = True,
            showlakes = True,
            landcolor = 'rgb(204, 204, 204)',
            countrycolor = 'rgb(204, 204, 204)',
            lakecolor = 'rgb(255, 255, 255)',
            projection_type = "equirectangular",
            coastlinewidth = 2,
            lataxis = dict(
                range = [47, 54],
                showgrid = True,
                dtick = 10
            ),
            lonaxis = dict(
                range = [8, 16],
                showgrid = True,
                dtick = 20
            )
        )
    )
    fig.show()

vis2()
#### Ab hier so b und c

df = pd.read_csv('route2.csv', delimiter=";")

df["Latitude"] = df["Latitude"].apply(lambda x: float(x.replace(",", ".")))
df["Longitude"] = df["Longitude"].apply(lambda x: float(x.replace(",", ".")))

def haversine(lat_1, lat_2, lon_1, lon_2):
    R = 6371
    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def prepare_df():
    start = df['Ort'][0]
    all_paths = []
    path = []
    for row in df.iterrows():
        path.append(row)
        if pd.notnull(row[1]['Ort']) and row[1]['Ort'] != start:
            start = row[1]['Ort']
            all_paths.append(path)
            path = []

    return all_paths

def get_lengths():
    all_distances = []
    all_paths = prepare_df()
    for path in all_paths:
        distance = 0
        for i in range(len(path) - 1):
            lon_2 = np.radians(path[i][1]['Longitude'])
            lon_1 = np.radians(path[i + 1][1]['Longitude'])
            lat_2 = np.radians(path[i][1]['Latitude'])
            lat_1 = np.radians(path[i + 1][1]['Latitude'])
            distance += haversine(lat_1, lat_2, lon_1, lon_2)
        all_distances.append(distance)
    return all_distances

def get_direct_distances():
    distances = []
    df = pd.read_csv('route1.csv', delimiter=";")

    df["Latitude"] = df["Latitude"].apply(lambda x: float(x.replace(",", ".")))
    df["Longitude"] = df["Longitude"].apply(lambda x: float(x.replace(",", ".")))
    for i in range(len(df) - 1):
        lon_2 = np.radians(df['Longitude'][i])
        lon_1 = np.radians(df['Longitude'][i + 1])
        lat_2 = np.radians(df['Latitude'][i])
        lat_1 = np.radians(df['Latitude'][i + 1])
        distance = haversine(lat_1, lat_2, lon_1, lon_2)
        distances.append(distance)
    return distances


print('Non direct')
print(np.round(get_lengths(), 3))
print('direct')
print(np.round(get_direct_distances(), 3))