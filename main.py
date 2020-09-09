#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(4)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from tkinter import ttk
from tkinter import StringVar

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import time
#import tkk

class Application(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.fileSelect = ""
        self.parent.title("Predicción de Ventas")
        self.menubar = tk.Menu(parent)
        self.filemenu = tk.Menu(self.menubar, tearoff = 0)
        self.filemenu.add_command(label = "Cargar data", command = self.selectFile)

        self.filemenu.add_separator()
        self.filemenu.add_command(label = "Guardar reporte")
        self.filemenu.add_separator()
        self.filemenu.add_command(label = "Salir", command = root.quit)
        self.menubar.add_cascade(label = "Dataset", menu = self.filemenu)
        self.editmenu = tk.Menu(self.menubar, tearoff=0)

        self.editmenu.add_command(label = "Entrena Red Neuronal", command = self.entrenamiento)

        self.menubar.add_cascade(label = "Entrenamiento", menu = self.editmenu)

        self.helpmenu = tk.Menu(self.menubar, tearoff=0)
        self.helpmenu.add_command(label = "Acerca de")
        self.menubar.add_cascade(label = "Ayuda", menu = self.helpmenu)

        self.parent.config(menu = self.menubar)
        
        self.text1 = StringVar()
        self.text1.set('Cargar archivo .xlsx o .csv')

        self.texto1 = ttk.Label(self.parent, textvariable=self.text1)
        self.texto1.grid(row=0)

        self.lf = ttk.Labelframe(self.parent, text='Ventas')
        self.lf.grid(row=1, column=0, sticky='nwes', padx=3, pady=3)

        '''
        t = np.arange(0.0,3.0,0.01)
        df = pd.DataFrame({'t':t, 's':((2*t/3)*np.pi+120)})

        fig = Figure(figsize=(5,4), dpi=100)
        ax = fig.add_subplot(111)

        df.plot(x='t', y='s', ax=ax)

        self.canvas = FigureCanvasTkAgg(fig, master=lf)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)
        '''

    def graficar_predicciones(self, real, prediccion):
        
        
        #fig = Figure(figsize=(5,4), dpi=100)

        fig = plt.figure(figsize=(4, 5))
        plt.plot(real[0:len(prediccion)],color='red', label='Valor real')
        plt.plot(prediccion, color='blue', label='Predicción')
        plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
        plt.xlabel('Tiempo')
        plt.ylabel('Ingresos')
        plt.legend()
        #plt.show()
        canvas = FigureCanvasTkAgg(fig, master=self.lf)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=1)

    def selectFile(self):
        fname = askopenfilename(filetypes=(("Archivo Dataset", "*.csv"),
                                           ("Hojas de calculo", "*.xlsx;*.xls"),
                                           ("Todos los archivos", "*.*") ))
        if fname:
            try:
                self.text1.set(fname)
                self.fileSelect = fname
            except:
                print("")


    def entrenamiento(self):
        print("archivo a entrenar: "+self.fileSelect)
        dataset = pd.read_csv(self.fileSelect, index_col='Date', parse_dates=['Date'])
        dataset.head()
        set_entrenamiento = dataset[:'2019'].iloc[:,0:1]
        set_validacion = dataset['2020'].iloc[:,0:1]

        set_entrenamiento['Cantidad'].plot(legend=True)
        set_validacion['Cantidad'].plot(legend=True)

        fig = plt.figure(figsize=(5, 5))
        plt.plot(set_entrenamiento['Cantidad'])
        plt.plot(set_validacion['Cantidad'])
        plt.legend(['Entrenamiento (2018-2019)', 'Validación (2020)'])
        canvas = FigureCanvasTkAgg(fig, master=self.lf)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0)

        sc = MinMaxScaler(feature_range=(0,1))
        set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

        time_step = 60
        X_train = []
        Y_train = []
        m = len(set_entrenamiento_escalado)

        for i in range(time_step,m):
            X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
            Y_train.append(set_entrenamiento_escalado[i,0])
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        dim_entrada = (X_train.shape[1],1)
        dim_salida = 1
        na = 50

        modelo = Sequential()
        modelo.add(LSTM(units=na, input_shape=dim_entrada))
        modelo.add(Dense(units=dim_salida))
        modelo.compile(optimizer='rmsprop', loss='mse')
        modelo.fit(X_train,Y_train,epochs=20,batch_size=32)

        x_test = set_validacion.values
        x_test = sc.transform(x_test)

        X_test = []
        for i in range(time_step,len(x_test)):
            X_test.append(x_test[i-time_step:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        prediccion = modelo.predict(X_test)
        prediccion = sc.inverse_transform(prediccion)

        self.graficar_predicciones(set_validacion.values,prediccion)




if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x500")
    Application(root)#.pack(side="top", fill="both", expand=True)
    root.mainloop()
