import altair as alt
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.palettes import color_palette
from colormap import rgb2hex

from datetime import datetime as dt
from datetime import timedelta as td
df = pd.read_csv('Infovis/temperatures.csv',delimiter=";")
df["Datum"] = df["Datum"].apply(lambda x: dt.strptime(x,'%d.%m.%Y'))
df["Temperatur"] = df["Temperatur"].apply(lambda x: int(x))

area1 = alt.Chart(df).mark_area(
    clip=True,
    interpolate='monotone'
).encode(
    alt.X('Datum',scale=alt.Scale(zero=False, nice=True)),
    alt.Y('Temperatur', scale=alt.Scale(domain=[-20, 20]), title='Temperatur'),
    opacity=alt.value(0.6)
).properties(
    width=300,
    height=150
)

area3 = area1.mark_area(color='rgba(255,99,71,0.5)', clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,5])),
).transform_calculate(
    "Temperatur", alt.datum.Temperatur
)
area4 = area1.mark_area(color='rgba(255,0,0,0.5)', clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,5])),
).transform_calculate(
    "Temperatur", alt.datum.Temperatur-5
)
area5 = area1.mark_area(color='rgba(220,20,60,0.5)', clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,5])), 
).transform_calculate(
    "Temperatur", alt.datum.Temperatur-10
)
area6 = area1.mark_area(color='rgba(178,34,34,0.5)', clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,5])),
).transform_calculate(
    "Temperatur", alt.datum.Temperatur-15
)
area1 = area1.mark_area(color="#5dade2", clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,-5]))
).transform_calculate(
    "Temperatur", alt.datum.Temperatur
)
area =area3+area4+area5+area6
area.show()
sns.barplot(x=df["Datum"],y=df["Temperatur"])
handles = [              Patch(facecolor=str(rgb2hex(178,34,34)),label='15° bis 20°'),
                         Patch(facecolor=str(rgb2hex(220,20,60)), label='10° bis 15°'),
                         Patch(facecolor=str(rgb2hex(220,20,60)), label='5° bis 10°'),
                         Patch(facecolor=str(rgb2hex(255,0,0)), label='0° bis 5°'),
                         Patch(facecolor=str(rgb2hex(158,204,236)), label='0° bis -5°'),
                         Patch(facecolor=str(rgb2hex(102,176,225)), label='-5° bis -10°'),
                         Patch(facecolor=str(rgb2hex(40,140,206)), label='-10° bis -15°'),
                         Patch(facecolor=str(rgb2hex(30,104,153)), label='-15° bis -20°')]
plt.legend(handles=handles,bbox_to_anchor=(1, 0.5))
plt.show() 


import plotly.graph_objects as go

a = '{1, 2, 3, 4, 5, 6} \n \n 100'
b = '{1, 2} \n \n 30'
c = '{3, 4, 5, 6} \n \n 70'
e = '{3, 4} \n \n 36'
d = "{5, 6} \n \n 34"
f = "{5} \n \n 16"
g = "{6} \n \n 18"
test = "test"
temp = "temp"

labels = [test, a, temp, b, c, d, e, f, g]
parents = ["", test, test, a, a, c, c, d, d]
values = [200, 100, 100, 30, 70, 36, 34, 16, 18]

fig = go.Figure(go.Treemap(
    branchvalues="total",
    labels=labels,
    parents=parents,
    values=values,
    textinfo="label",
))
fig.update_traces(
    root_color="lightgrey",
    textfont={"size":20})
fig.layout.hovermode = False
fig.update_layout(margin = dict(t=150, l=250, r=250, b=50))
fig.show()

import plotly.express as px

labels = [ a, b, c, d, e, f, g]
data = dict(
    character=["{1, 2, 3, 4, 5, 6}", "{1, 2}", "{3, 4 ,5, 6}", "{3, 4}", "{5, 6}", "{5}", "{6}"],
    parent=["", "{1, 2, 3, 4, 5, 6}", "{1, 2, 3, 4, 5, 6}", "{3, 4 ,5, 6}", "{3, 4 ,5, 6}", "{5, 6}", "{5, 6}"],
    value=[100, 30, 70, 36, 34, 16, 18])

fig =px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
    labels=data["value"]
)
fig.layout.hovermode=False
fig.show()