import altair as alt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta as td
df = pd.read_csv('temperatures.csv',delimiter=";")
df["Datum"] = df["Datum"].apply(lambda x: dt.strptime(x,'%d.%m.%Y'))
df["Temperatur"] = df["Temperatur"].apply(lambda x: int(x))
print(df)
area1 = alt.Chart(df).mark_area(
    clip=True,
    interpolate='monotone'
).encode(
    alt.X('Datum', scale=alt.Scale(zero=False, nice=False)),
    alt.Y('Temperatur', scale=alt.Scale(domain=[-20, 20]), title='Temperatur'),
    opacity=alt.value(1)
).properties(
    width=500,
    height=75
)
area2 = area1.encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[-10, 0])),opacity=alt.value(0.6)
)
area3 = area1.encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0, 10]))
)
area4 = area1.encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[10, 20]))
)
area1 = area1 + area2 + area3 + area4
area1.show()