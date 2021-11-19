import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib.patches as patches
from seaborn.palettes import color_palette


""" Homework 4.1 """
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


""" Homework 4.2 """

df = pd.read_csv("landslides.csv", delimiter=";")
df["year"] = df["date"].apply(lambda x: str(dt.strptime(x,'%d.%m.%Y').strftime("%Y")))
df["date"] = df["date"].apply(lambda x: dt.strptime(x,'%d.%m.%Y'))
df["lat"] = df["geolocation"].apply(lambda x:float(x.split(",")[0][1:]) )
df["lon"] = df["geolocation"].apply(lambda x:float(x.split(",")[1][:-1]) )

def transform(x):
    try:
        return int(x.split("_")[2].strip("km"))
    except Exception as e:
        pass

df["radius"] = df["location_accuracy"].apply(lambda x: transform(x))

df = df[df.year.isin(["2011","2015"])]
df = df.sort_values("date")
df["dif_lat"] = df["lat"].diff()
df["dif_lon"] = df["lon"].diff()

df = df[["year","date","radius","dif_lat","dif_lon","countrycode"]]
df = df.sort_values(by=["year","countrycode","date"])
#print(df.where(df["countrycode"]=="US").where(df["year"]=="2011").dropna().tail(50))

lastDate = dt.now()
lastRadius = 0.0
lastCountry = ""
a = 2

# Filter rows
for index, row in df.iterrows():
    if (row["date"] - lastDate).days < 3:
        if lastRadius != "nan" or row["radius"] != "nan":
            c = np.sqrt(row["dif_lat"]**2  +  row["dif_lon"]**2)
            if (c < row["radius"] or c < lastRadius):
                df.drop(index, inplace=True)
            elif np.sqrt(row["dif_lat"]**2  +  row["dif_lon"]**2) <= float(a):
                df.drop(index, inplace=True)
    lastDate = row["date"]
    lastRadius = row["radius"]
    lastCountry = row["countrycode"]

df_count = pd.DataFrame({"year":["2011","2015"], "amount": [list(df[df["year"]=="2011"].count())[0],list(df[df["year"]=="2015"].count())[0]]})
print(df_count)

b = sns.countplot(x="year",data=df)
# change bar size
for patch in b.patches :
        current_width = patch.get_width()
        diff = current_width - .35
        # we change the bar width
        patch.set_width(.35)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

plt.legend(loc="upper left", handles=[patches.Rectangle((0,0),1,1, color='cornflowerblue'),patches.Rectangle((0,0),1,1, color='sandybrown')], labels=[str(list(df_count["amount"])[0]),str(list(df_count["amount"])[1])])
plt.xlabel("Jahr")
plt.ylabel("Anzahl")
plt.title("Registrierte Erdrutsche (Radius <= "+str(a)+" km)")
plt.show()