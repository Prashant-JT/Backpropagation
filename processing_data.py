import pandas as pd
import numpy as np


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

    # df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA']).dt.tz_convert(None)
    df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA'], format='%Y-%m-%d %H:%M:%S')
    df['HORA'] = df['FECHA_HORA'].dt.hour
    df['DIA'] = df['FECHA_HORA'].dt.day
    df['MES'] = df['FECHA_HORA'].dt.month

# Método no empleado para duplicar los "1" y balancear los datos.
def duplicate(df):
    rep = df[df['ACCIDENTE'] == 1]
    return df.append([rep] * 18, ignore_index=True)


def load_train():
    df = load_dataset(r'dataset.xls').replace(r'^\s*$', np.nan, regex=True).dropna()
    df['ACCIDENTE'] = df['ACCIDENTE'].replace(to_replace=['No', 'Yes'], value=[0, 1])

    aux = df['ACCIDENTE']
    change_date_format(df)
    df.drop(columns=['FECHA_HORA', 'ACCIDENTE'], inplace=True)
    df = pd.get_dummies(df, columns=['TIPO_PRECIPITACION', 'INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'])
    df['ACCIDENTE'] = aux

    df = df.astype(dtype='float64')

    dfX_test = df.sample(frac=0.05)
    df.drop(dfX_test.index[:], inplace=True)
    df.reset_index(drop=True, inplace=True)
    dfX_test.reset_index(drop=True, inplace=True)
    dfX_test.to_excel("test.xls", index=False)

    x_train = df.iloc[:, 0:23].values
    y_train = df.iloc[:, 23].values

    return x_train, y_train


def load_test():
    dfX_test = load_dataset(r'test.xls')
    y_test = dfX_test.iloc[:, 23].values
    x_test = dfX_test.iloc[:, 0:23].values

    return x_test, y_test
