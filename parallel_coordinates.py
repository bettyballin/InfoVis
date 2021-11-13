import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

def remove_na(x):
    if "#VALUE!" in str(x) or "—" in str(x) or "NM" in str(x) or "DNF" in str(x) :
        return 0.0
    try:
        return float(str(x).replace(",","."))
    except Exception as e:
        pass
    m = str(x).split(":")
    seconds = float(m[0])*60
    if len(m) > 1:
        s = m[1].split(".")
        seconds += float(s[0])
        if len(s) > 1:
            seconds += float("0."+s[1])
    return seconds

def clean_data(df):
    columns = ['Competitor', '100m', 'Long_Jump', 'Shot_Put', 'High_Jump', '400', '110m_Hurdles', 'Discus_Throw', 'Pole_Vault', 'Javelin_Throw', '1500m']
    columns = ['Competitor'] + [x + '_pts' for x in columns[1:]]
    competitors = ["Attila Zsivóczky-Pandel", "Dmitriy Karpov", "Romain Barras", "Roman Šebrle", "Yordani García"]
    df = df[columns]
    df = df.groupby("Competitor", as_index=False).mean()
    df = df.loc[df['Competitor'].isin(competitors)]
    return df


def load_decathon_data():
    df = pd.read_csv("decathlon.csv", delimiter=",")
    for c in ["100m", "Long_Jump", "Shot_Put", "High_Jump", "400m", "110m_Hurdles", "Discus_Throw", "Pole_Vault",
              "Javelin_Throw", "1500m"]:
        df[c] = df[c].apply(lambda x: remove_na(x))
    df = clean_data(df)
    return df

cleaned_data = load_decathon_data()

print(cleaned_data.head())


def plot_parallel(df):
    columns = ['Competitor', '100m_pts', 'Long_Jump_pts', 'Shot_Put_pts', 'High_Jump_pts', '400_pts', '110m_Hurdles_pts',
        'Discus_Throw_pts', 'Pole_Vault_pts', 'Javelin_Throw_pts', '1500m_pts']
    new_column_names = ['100m', 'Long Jump', 'Shot Put', 'High Jump', '400m', '110m Hurdles', 'Discus Throw', 'Pole Vault', 'Javelin Throw', '1500m']
    colname_mapping = {
        '100m_pts': '100m',
        'Long_Jump_pts': 'Long Jump',
        'Shot_Put_pts': 'Shot Put',
        'High_Jump_pts': 'High Jump',
        '400_pts': '400m',
        '110m_Hurdles_pts': '110m Hurdles',
        'Discus_Throw_pts': 'Discus Throw',
        'Pole_Vault_pts': 'Pole Vault',
        'Javelin_Throw_pts': 'Javelin Throw',
        '1500m_pts': '1500m'
    }
    df.rename(columns=colname_mapping, inplace=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'purple']
    parallel_coordinates(df, 'Competitor', color=colors)
    plt.title(r'$\bf{Durchschnittliche \ Punktzahl \ Zehnkampfdisziplinen}$'
    '\n'
    r'Quelle: decathlon.csv')
    plt.xlabel('Disziplin')
    plt.ylabel('Gemittelte Punktzahl')
    plt.legend(title=r'$\bf{Teilnehmer}$')
    plt.show()

plot_parallel(cleaned_data)