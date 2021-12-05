import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import plotly.express as px

def task1():
    df = pd.read_csv('abc.csv')

    df.reset_index(level=0, inplace=True)
    print(df)

    test_df = pd.read_csv('https://raw.github.com/pandas-dev/pandas/master/pandas/tests/io/data/csv/iris.csv')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:cyan']

    pd.plotting.parallel_coordinates(df, 'index', color=colors)

    plt.xlabel('Dimension')
    plt.ylabel('Koordinate')
    plt.title('Parallele Koordinaten von 6 Datenpunkten \n Quelle: Abbildung 2')
    plt.legend(title='Datenpunkte')

    plt.show()

def task2():
    data = {'ID': [1, 2, 3, 4, 5, 6, 7, 8],
            'Geschlecht': ['M', 'M', 'M', 'M', 'W', 'W', 'W', 'W'],
            'Altersgruppe': ['ab 40', 'bis 39', 'ab 40', 'bis 39', 'ab 40', 'bis 39', 'ab 40', 'bis 39'],
            'Führerschein': ['ja', 'ja', 'nein', 'nein', 'ja', 'ja', 'nein', 'nein'],
            'Anzahl Menschen': [25, 18, 0, 7, 18, 12, 14, 6]}

    df = pd.DataFrame(data)

    colors = 25 * ['#d25304'] + 18 * ['#ff9f7a'] + 7 * ['#ff9f7a'] + 18 * ['#2390fd'] + 12 * ['#87cdf8'] + 14 * ['#2390fd'] + 6 * ['#87cdf8']

    result_df = pd.DataFrame()
    for row in df.iterrows():
        for i in range(row[1]['Anzahl Menschen']):
            result_df = result_df.append(row[1])



    print(result_df)
    fig = px.parallel_categories(result_df, dimensions=['Geschlecht', 'Führerschein', 'Altersgruppe'], color=colors)
    fig.update_layout(
        title="Parallel Set Plot von Führerscheindaten \n Quelle: Tabelle 1",
        font=dict(
            size=30
        )
    )
    fig.show()
task2()
