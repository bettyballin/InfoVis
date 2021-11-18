import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td

from seaborn.palettes import color_palette

df = pd.read_csv("landslides.csv", delimiter=";")
df["year"] = df["date"].apply(lambda x: str(dt.strptime(x,'%d.%m.%Y').strftime("%Y")))
df = df[df.year.isin(["2011","2015"])]
print(len(df.where(df["year"]  == "2011")))
print(len(df.where(df["year"]  == "2015")))
#print(df.head(5))

df_cars = pd.read_csv("cars.csv", delimiter=";")
df_cars["ccm"] = df_cars["ccm"].apply(lambda x: x/10)
df_cars["doors"] = df_cars["doors"].apply(lambda x: x*10)
print(df_cars.head(6))
print(list(range(6)))
a = sns.barplot(data=df_cars,x=list(range(6)),y="kW",hue="company")
plt.plot(list(range(6)),df_cars["ccm"],color="black",label="ccm")
#sns.pointplot(data=df_cars, x=list(range(6)),y="doors")
h1,l1 = a.get_legend_handles_labels()
plt.legend(h1,l1)
plt.show()


'''
sns.set_style("ticks")
sns.set_style("dark")

df = df[df["Competitor"].isin(["Roman Šebrle", "Romain Barras", "Yordani García", "Dmitriy Karpov", "Attila Zsivóczky-Pandel"])]
df = df[["Competitor","100m","Long_Jump","Shot_Put","High_Jump","400m","110m_Hurdles","Discus_Throw","Pole_Vault","Javelin_Throw","1500m"]]
df = df.dropna()
print(len(df))

for c in ["100m","Long_Jump","Shot_Put","High_Jump","400m","110m_Hurdles","Discus_Throw","Pole_Vault","Javelin_Throw","1500m"]:
    df[c] = df[c].apply(lambda x: get_seconds(x)  )
print(len(df))

df = df.groupby("Competitor",as_index=False).mean()
df = pd.melt(df, id_vars=['Competitor'], value_vars=df.columns[1:],var_name="Discipline",value_name="Seconds")
#df = df.sort_values(by="Discipline")

df_max = df.loc[df.reset_index().groupby(['Discipline'])['Seconds'].idxmax()]
df_max = df_max.sort_values("Discipline")
df_min = df.loc[df.reset_index().groupby(['Discipline'])['Seconds'].idxmin()]
df_min = df_min.sort_values("Discipline")

competitors = []
disciplines = []
percentiles = []

for d in df.values:
    if "m" in d:
        c = df_min[df_min["Discipline"] == d[1]].astype(str)
    else:
        c = df_max[df_max["Discipline"] == d[1]]
    
    percentiles.append(d[2]/c["Seconds"].iloc[0])
    competitors.append(c["Competitor"].iloc[0])
    disciplines.append(c["Discipline"].iloc[0])

print(df.head(60))
cs = list(df["Competitor"].iloc[:])
print([int(c) for c in cs])
p = pd.DataFrame({"Competitors":["Attila Zsivóczky-Pandel", "Dmitriy Karpov", "Romain Barras", "Roman Šebrle", "Yordani García"]*10, "Disciplines":disciplines, "Percentile":percentiles})
print(p.head(50))
fig = parallel_coordinates( df, color=pd.to_numeric(df["Competitor"].iloc[:]), labels="Discipline")
fig.show()
#g = sns.barplot(data=p,x="Disciplines", y="Percentile",  hue="Competitors")
#g.set_ylim(0.777,1.01)

#ax = sns.barplot(data=df, x="Discipline", y="Seconds", hue="Competitor")
#ax.grid()
"""
heights = [0,0,0,0,0,0,0,0,0,0]
patches = [0,0,0,0,0,0,0,0,0,0]

for patch in ax.patches:
    if patch.xy[0] < 0.5:
        i = 0
    elif patch.xy[0] < 1.5:
        i = 1
    elif patch.xy[0] < 2.5:
        i = 2
    elif patch.xy[0] < 3.5:
        i = 3
    elif patch.xy[0] < 4.5:
        i = 4
    elif patch.xy[0] < 5.5:
        i = 5
    elif patch.xy[0] < 6.5:
        i = 6
    elif patch.xy[0] < 7.5:
        i = 7
    elif patch.xy[0] < 8.5:
        i = 8
    else:
        i = 9
    h = patch.get_height()
    if heights[i] < h:
        heights[i] = h
        patches[i] = patch
        
for patch in patches:
    ax.text(patch.get_x()+patch.get_width()/2.,
    patch.get_height()*(1.01),
    str("{:.1f}".format(float(str(patch.get_height())))),
    ha = 'center'  )
#sns.scatterplot(data=df_n, x=[0,0.8,2,3,4,5,6,7,8,9], y="Seconds", marker="*", color="k", s=200, zorder=10, legend=False)
"""
plt.legend(loc="lower left")
plt.tight_layout()
plt.title("Mittlerer Leistungswert pro Disziplin und Sportler")
labels, locations = plt.xticks()
plt.xticks(labels, locations, rotation='15')
plt.show()

"""
df["Datum"] = df["Datum"].apply(lambda x: dt.strptime(x,'%Y-%m-%d').strftime('%d'))
print(df.head(10))
df.sort_values(by="Stadt").aggregate("Sonnenschein")
df_N = df.groupby(['Stadt'], sort=False)['Niederschlag'].max()
df["Sonnenschein"]= df["Sonnenschein"].where(df["Sonnenschein"] != "-" )
df["Sonnenschein"] = df["Sonnenschein"].apply(lambda x: float(x))
df_S = df.groupby(['Stadt'], sort=False)['Sonnenschein'].max()
df = pd.melt(df, id_vars=['Stadt','Datum'], value_vars=['Niederschlag', 'Sonnenschein'])
a = sns.scatterplot(x=[-0.2, 0.8, 1.8, 2.8, 3.8], y=df_N, s=90,marker="^")
b = sns.scatterplot(x=[0.2, 1.2, 2.2, 3.2, 4.2], y=df_S, s=90,marker="^")
c = sns.scatterplot(x=[1.2], y=[1], color=".2", s=90, marker="X")
g = sns.boxplot(x="Stadt", y="value",hue="variable",data=df, fliersize=0)
h3,l3 = g.get_legend_handles_labels()
#sns.despine(offset=10, trim=True)
plt.xlabel("")
plt.ylabel("Niederschlag (ml), Sonnenstunden (h)")
h1, = plt.plot([], [], '^', label="Höchste Niederschlagsmenge")
h2, =plt.plot([], [], '^', label="Meiste Sonnenstunden")
h3, =plt.plot([], [], 'X', color=".2", label="Fehlende Daten")

blue_patch = mpatches.Patch(color='lightsteelblue', label='')
orange_patch = mpatches.Patch(color='sandybrown', label='')
plt.legend([h1,h2,h3, blue_patch,orange_patch],["Höchste Niederschlagsmenge","Meiste Sonnenstunden","Fehlende Daten","Verteilung Niederschlagsmenge","Verteilung Sonnenstunden"])

    if c == "100m":
        df[c] = df[c].apply(lambda x: 100/x )
    if c == "110m_Hurdles":
        df[c] = df[c].apply(lambda x: 110/x )
    if c == "1500m":
        df[c] = df[c].apply(lambda x: 1500/x )
    if c == "400m":
        df[c] = df[c].apply(lambda x: 400/x )
        
"""
'''