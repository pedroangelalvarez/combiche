#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
import numpy as np
np.random.seed(4)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
'''
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('fast')

from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, LSTM
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam


import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from tkinter import ttk
from tkinter import HORIZONTAL
from tkinter import StringVar, Label, Button
from tkcalendar import Calendar, DateEntry

import time
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import openpyxl
import xlsxwriter
import os.path
from os import path
from shutil import copyfile

def agregarNuevoValor(x_test,nuevoValor,ultDiaSemana):
    for i in range(x_test.shape[2]-3):
        x_test[0][0][i+2] = x_test[0][0][i+3]
    ultDiaSemana=ultDiaSemana+1
    if ultDiaSemana>6:
        ultDiaSemana=0
    x_test[0][0][0]=ultDiaSemana
    x_test[0][0][1]=12
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test,ultDiaSemana

PASOS = 7
daterange = pd.date_range('2020-01-02', '2020-02-02')
for fecha in daterange:
    fechaIni = str(fecha - datetime.timedelta(days=15))
    fechaFin = str(fecha + datetime.timedelta(days=1))
    #fechaFin = str(cal2.get_date())
    print(fechaFin)
    print(fechaIni)
    df = pd.read_csv('temp.csv',  parse_dates=[0], header=None,index_col=0, names=['fecha','unidades'])
    df['weekday']=[x.weekday() for x in df.index]
    df['month']=[x.month for x in df.index]
    ultimosDias = df[fechaIni:fechaFin]
    print("ULTIMOS DIAS")
    print(ultimosDias)
    values = ultimosDias['unidades'].values

    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))

    values=values.reshape(-1, 1) 
    scaled = scaler.fit_transform(values)

    reframed = series_to_supervised(scaled, PASOS, 1)
    reframed.reset_index(inplace=True, drop=True)

    contador=0
    reframed['weekday']=ultimosDias['weekday']
    reframed['month']=ultimosDias['month']

    for i in range(reframed.index[0],reframed.index[-1]):
        reframed['weekday'].loc[contador]=ultimosDias['weekday'][i+8]
        reframed['month'].loc[contador]=ultimosDias['month'][i+8]
        contador=contador+1
    reframed.head()

    reordenado=reframed[ ['weekday','month','var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)'] ]
    reordenado.dropna(inplace=True)
    values = reordenado.values
    x_test = values[5:, :]
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))


    ultDiaSemana = reordenado.weekday[len(reordenado)-1]

    model = load_model('model.h5')

    results = []
    for i in range(7):
        dia=np.array([x_test[0][0][0]])
        mes=np.array([x_test[0][0][1]])
        valores=np.array([x_test[0][0][2:9]])
        parcial=model.predict([dia, mes, valores])
        results.append(parcial[0])
        print('pred',i,x_test)
        x_test,ultDiaSemana=agregarNuevoValor(x_test,parcial[0],ultDiaSemana)

    adimen = [x for x in results]
    inverted = scaler.inverse_transform(adimen)
    #inverted


    prediccionProxSemana = pd.DataFrame(inverted)
    prediccionProxSemana.columns = ['pronostico']
    prediccionProxSemana.plot()
    prediccionProxSemana.to_csv('pronostico.csv')

    predicciones = prediccionProxSemana.to_numpy()
    y_values=[]
    for i in range(len(predicciones)):
        y_values.append(predicciones[i][0])

    x_values = []
    for i in range(len(predicciones)):
        x_values.append(datetime.datetime.strptime(str(cal1.get_date() + datetime.timedelta(days=i)),"%Y-%m-%d").date())

