import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.graphics.mosaicplot import mosaic

data = {'ID': [ 5, 6, 7, 8, 1, 2, 3, 4,],
       'Geschlecht': ['W', 'W', 'W', 'W','M', 'M', 'M', 'M'],
       'Altersgruppe': ['ab 40', 'bis 39', 'ab 40', 'bis 39','ab 40', 'bis 39', 'ab 40', 'bis 39'],
       'Führerschein': ['Führerschein', 'Führerschein', 'kein Führerschein', 'kein Führerschein','Führerschein', 'Führerschein', 'kein Führerschein', 'kein Führerschein'],
       'Anzahl': [18, 12, 14, 6,25, 18, 0, 7]}

df = pd.DataFrame(data)
df = df.loc[df.index.repeat(df.Anzahl)]
props={}
props[('Führerschein','M','ab 40')]={'facecolor':'darkorange', 'edgecolor':'white'}
props[('Führerschein','M','bis 39')]={'facecolor':'lightsalmon', 'edgecolor':'white'}
props[('kein Führerschein','M','ab 40')]={'facecolor':'#ab765a', 'edgecolor':'white'}
props[('kein Führerschein','M','bis 39')]={'facecolor':'lightsalmon', 'edgecolor':'white'}
props[('Führerschein','W','ab 40')]={'facecolor':'dodgerblue','edgecolor':'white'}
props[('Führerschein','W','bis 39')]={'facecolor':'lightskyblue','edgecolor':'white'}
props[('kein Führerschein','W','ab 40')]={'facecolor':'dodgerblue','edgecolor':'white'}
props[('kein Führerschein','W','bis 39')]={'facecolor':'lightskyblue','edgecolor':'white'}
labelizer=lambda k:{('Führerschein','M','ab 40'):25,('Führerschein','M','bis 39'):18,('kein Führerschein','M','ab 40'):"",('kein Führerschein','M','bis 39'):7,('Führerschein','W','ab 40'):18,('Führerschein','W','bis 39'):12,('kein Führerschein','W','ab 40'):14,('kein Führerschein','W','bis 39'):6}[k]
mosaic(df, [ 'Führerschein','Geschlecht','Altersgruppe'],title="Umfrageergebnisse im Mosaic Plot",properties=props, labelizer=labelizer)

p1 = mpatches.Patch(color='darkorange', label='Männlich, ab 40')
p2 = mpatches.Patch(color='lightsalmon', label='Männlich, unter 39')
p3 = mpatches.Patch(color='dodgerblue', label='Weiblich, ab 40')
p4 = mpatches.Patch(color='lightskyblue', label='Weiblich, unter 39')
plt.legend(handles=[p1,p2,p3,p4], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
