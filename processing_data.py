import pandas as pd
import numpy as np


def load_dataset(name):
    df = pd.read_excel(name)
    return df


def load_train():
    df = load_dataset(r'dataset.xls').replace(r'^\s*$', np.nan, regex=True).dropna()
    df['ACCIDENTE'] = df['ACCIDENTE'].replace(to_replace=['No', 'Yes'], value=[0, 1])

    aux = df['ACCIDENTE']
    df.drop(columns=['FECHA_HORA', 'ACCIDENTE'], inplace=True)
    df = pd.get_dummies(df, columns=['TIPO_PRECIPITACION', 'INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'])
    df['ACCIDENTE'] = aux

    df = df.astype(float)

    dfX_test = df.sample(frac=0.05)
    df.drop(dfX_test.index[:], inplace=True)
    df.reset_index(drop=True, inplace=True)  # Creo que esto no hace falta
    dfX_test.reset_index(drop=True, inplace=True)  # Igual que esto tampoco
    dfX_test.to_excel("test.xls", index=False)

    x_train = df.iloc[:, 0:20].values
    y_train = df.iloc[:, 20].values

    return x_train, y_train


def load_test():
    dfX_test = load_dataset(r'test.xls')
    y_test = dfX_test.iloc[:, 20].values
    x_test = dfX_test.iloc[:, 0:20].values

    return x_test, y_test
