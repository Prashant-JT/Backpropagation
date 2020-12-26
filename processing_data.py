import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date

def pie_accidents_monthly(df):
    january = df.loc[(df['MES'] == 1) & (df['ACCIDENTE'] == 1)]
    february = df.loc[(df['MES'] == 2) & (df['ACCIDENTE'] == 1)]
    march = df.loc[(df['MES'] == 3) & (df['ACCIDENTE'] == 1)]
    april = df.loc[(df['MES'] == 4) & (df['ACCIDENTE'] == 1)]
    may = df.loc[(df['MES'] == 5) & (df['ACCIDENTE'] == 1)]
    june = df.loc[(df['MES'] == 6) & (df['ACCIDENTE'] == 1)]
    july = df.loc[(df['MES'] == 7) & (df['ACCIDENTE'] == 1)]
    august = df.loc[(df['MES'] == 8) & (df['ACCIDENTE'] == 1)]
    september = df.loc[(df['MES'] == 9) & (df['ACCIDENTE'] == 1)]
    october = df.loc[(df['MES'] == 10) & (df['ACCIDENTE'] == 1)]
    november = df.loc[(df['MES'] == 11) & (df['ACCIDENTE'] == 1)]
    december = df.loc[(df['MES'] == 12) & (df['ACCIDENTE'] == 1)]

    x = [len(january), len(february), len(march), len(april), len(may), len(june), len(july),
         len(august), len(september), len(october), len(november), len(december)]

    my_xticks = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                 "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

    plt.pie(x, autopct='%1.1f%%', shadow=True, pctdistance=1.2)
    plt.legend(my_xticks, loc='best', bbox_to_anchor=(0.20, 0.60))

    plt.show()


def pie_type_accidents(df):
    clear = df[(df['TIPO_PRECIPITACION'] == "clear") & (df['ACCIDENTE'] == 1)]
    rain = df[(df['TIPO_PRECIPITACION'] == "rain") & (df['ACCIDENTE'] == 1)]
    snow = df[(df['TIPO_PRECIPITACION'] == "snow") & (df['ACCIDENTE'] == 1)]

    x = [len(clear), len(rain), len(snow)]

    my_xticks = ["Despejado", "Lluvia", "Nieve"]

    plt.pie(x, autopct='%1.1f%%', shadow=True, pctdistance=1.2)
    plt.legend(my_xticks, loc='best', bbox_to_anchor=(0.10, 0.60))

    plt.show()


def ploy_monthly_accidents(df):
    one = df.loc[(df['DIA'] == 1) & (df['ACCIDENTE'] == 1)]
    two = df.loc[(df['DIA'] == 2) & (df['ACCIDENTE'] == 1)]
    three = df.loc[(df['DIA'] == 3) & (df['ACCIDENTE'] == 1)]
    four = df.loc[(df['DIA'] == 4) & (df['ACCIDENTE'] == 1)]
    five = df.loc[(df['DIA'] == 5) & (df['ACCIDENTE'] == 1)]
    six = df.loc[(df['DIA'] == 6) & (df['ACCIDENTE'] == 1)]
    seven = df.loc[(df['DIA'] == 7) & (df['ACCIDENTE'] == 1)]
    eight = df.loc[(df['DIA'] == 8) & (df['ACCIDENTE'] == 1)]
    nine = df.loc[(df['DIA'] == 9) & (df['ACCIDENTE'] == 1)]
    ten = df.loc[(df['DIA'] == 10) & (df['ACCIDENTE'] == 1)]
    eleven = df.loc[(df['DIA'] == 11) & (df['ACCIDENTE'] == 1)]
    twelve = df.loc[(df['DIA'] == 12) & (df['ACCIDENTE'] == 1)]
    thirteen = df.loc[(df['DIA'] == 13) & (df['ACCIDENTE'] == 1)]
    fourteen = df.loc[(df['DIA'] == 14) & (df['ACCIDENTE'] == 1)]
    fiveteen = df.loc[(df['DIA'] == 15) & (df['ACCIDENTE'] == 1)]
    sixteen = df.loc[(df['DIA'] == 16) & (df['ACCIDENTE'] == 1)]
    seventeen = df.loc[(df['DIA'] == 17) & (df['ACCIDENTE'] == 1)]
    eighteen = df.loc[(df['DIA'] == 18) & (df['ACCIDENTE'] == 1)]
    nineteen = df.loc[(df['DIA'] == 19) & (df['ACCIDENTE'] == 1)]
    twenty = df.loc[(df['DIA'] == 20) & (df['ACCIDENTE'] == 1)]
    twentyone = df.loc[(df['DIA'] == 21) & (df['ACCIDENTE'] == 1)]
    twentytwo = df.loc[(df['DIA'] == 22) & (df['ACCIDENTE'] == 1)]
    twentythree = df.loc[(df['DIA'] == 23) & (df['ACCIDENTE'] == 1)]
    twentyfour = df.loc[(df['DIA'] == 24) & (df['ACCIDENTE'] == 1)]
    twentyfive = df.loc[(df['DIA'] == 25) & (df['ACCIDENTE'] == 1)]
    twentysix = df.loc[(df['DIA'] == 26) & (df['ACCIDENTE'] == 1)]
    twentyseven = df.loc[(df['DIA'] == 27) & (df['ACCIDENTE'] == 1)]
    twentyeight = df.loc[(df['DIA'] == 28) & (df['ACCIDENTE'] == 1)]
    twentynine = df.loc[(df['DIA'] == 29) & (df['ACCIDENTE'] == 1)]
    thirty = df.loc[(df['DIA'] == 30) & (df['ACCIDENTE'] == 1)]
    thirtyone = df.loc[(df['DIA'] == 31) & (df['ACCIDENTE'] == 1)]

    x = [len(one), len(two), len(three), len(four), len(five),
         len(six), len(seven), len(eight), len(nine), len(ten),
         len(eleven), len(twelve), len(thirteen), len(fourteen), len(fiveteen), len(sixteen),
         len(seventeen), len(eighteen), len(nineteen), len(twenty), len(twentyone),
         len(twentytwo), len(twentythree), len(twentyfour), len(twentyfive),
         len(twentysix), len(twentyseven), len(twentyeight), len(twentynine),
         len(thirty), len(thirtyone)]

    my_xticks = ["Día 1", "Día 2", "Día 3", "Día 4", "Día 5", "Día 6", "Día 7",
                 "Día 8", "Día 9", "Día 10", "Día 11", "Día 12", "Día 13", "Día 14",
                 "Día 15", "Día 16", "Día 17", "Día 18", "Día 19", "Día 20", "Día 21",
                 "Día 22", "Día 23", "Día 24", "Día 25", "Día 26", "Día 27", "Día 28",
                 "Día 29", "Día 30", "Día 31"]

    plt.plot(x)
    plt.xticks(range(len(my_xticks)), my_xticks, rotation=45)
    plt.xlabel('Días del mes')
    plt.ylabel('Número de accidentes')

    plt.show()


def load_dataset(name):
    df = pd.read_excel(name)
    return df


def change_date_format(df):
    f = []
    for index, row in df.iterrows():
        pos = row['FECHA_HORA'].find(",")
        if pos != -1:
            f.append(row['FECHA_HORA'][0:pos])
        else:
            df.drop(index, inplace=True)
    df['FECHA_HORA'] = f

    df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA'], format='%Y-%m-%d %H:%M:%S')
    df['HORA'] = df['FECHA_HORA'].dt.hour
    df['DIA'] = df['FECHA_HORA'].dt.day
    df['MES'] = df['FECHA_HORA'].dt.month

def load_train():
    df = load_dataset(r'dataset.xls').replace(r'^\s*$', np.nan, regex=True).dropna()
    df['ACCIDENTE'] = df['ACCIDENTE'].replace(to_replace=['No', 'Yes'], value=[0, 1])

    #pie_type_accidents(df)

    aux = df['ACCIDENTE']
    change_date_format(df)
    df.drop(columns=['FECHA_HORA', 'ACCIDENTE'], inplace=True)
    df = pd.get_dummies(df, columns=['TIPO_PRECIPITACION', 'INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'])
    df['ACCIDENTE'] = aux

    df = df.astype(float)

    #pie_accidents_monthly(df)
    ploy_monthly_accidents(df)

    dfX_test = df.sample(frac=0.05)
    df.drop(dfX_test.index[:], inplace=True)
    df.reset_index(drop=True, inplace=True)  # Creo que esto no hace falta
    dfX_test.reset_index(drop=True, inplace=True)  # Igual que esto tampoco
    dfX_test.to_excel("test.xls", index=False)

    #x_train = df.iloc[:, 0:23].values
    #y_train = df.iloc[:, 23].values

    #return x_train, y_train


def load_test():
    dfX_test = load_dataset(r'test.xls')
    y_test = dfX_test.iloc[:, 23].values
    x_test = dfX_test.iloc[:, 0:23].values

    return x_test, y_test