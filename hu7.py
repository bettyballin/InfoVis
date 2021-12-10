import altair as alt
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime as dt
from datetime import timedelta as td
df = pd.read_csv('temperatures.csv',delimiter=";")
df["Datum"] = df["Datum"].apply(lambda x: dt.strptime(x,'%d.%m.%Y'))
df["Temperatur"] = df["Temperatur"].apply(lambda x: int(x))

area1 = alt.Chart(df).mark_area(
    clip=True,
    interpolate='monotone'
).encode(
    alt.X('Datum', scale=alt.Scale(zero=False, nice=True)),
    alt.Y('Temperatur', scale=alt.Scale(domain=[-20, 20]), title='Temperatur'),
    opacity=alt.value(0.6)
).properties(
    width=300,
    height=200
)

area2 = area1.mark_area(color="#5dade2", clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,10]))
).transform_calculate(
    "Temperatur", alt.datum.Temperatur*(-1)
)
area3 = area1.mark_area(color='#e74c3c', clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,10]))
).transform_calculate(
    "Temperatur", alt.datum.Temperatur
)
area4 = area1.mark_area(color='#cb4335', clip=True).encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,10]))
).transform_calculate(
    "Temperatur", alt.datum.Temperatur-10
)
area1 = area1.encode(
    alt.Y('Temperatur:Q', scale=alt.Scale(domain=[0,10]))
).transform_calculate(
    Temperatur="datum.Temperatur-20"
)
#area2.show()
sns.barplot(x=df["Datum"],y=df["Temperatur"])
plt.legend(handles=[Patch(facecolor='#e7897f',
                         label='10° bis 20°'),Patch(facecolor='#f4b6b0',
                         label='0° bis 10°'),Patch(facecolor="#5dade2",
                         label='-10° bis 0°'),Patch(facecolor="#2e86c1",
                         label='-20° bis -10°')])
plt.show()