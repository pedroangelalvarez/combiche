#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(4)
import pandas as pd
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from tkinter import ttk

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

        ttk.Label(self.parent, text="Carga archivo .xlsx o .csv").grid(row=0)

        lf = ttk.Labelframe(self.parent, text='Ventas')
        lf.grid(row=1, column=0, sticky='nwes', padx=3, pady=3)


        t = np.arange(0.0,3.0,0.01)
        df = pd.DataFrame({'t':t, 's':((2*t/3)*np.pi+120)})

        fig = Figure(figsize=(5,4), dpi=100)
        ax = fig.add_subplot(111)

        df.plot(x='t', y='s', ax=ax)

        self.canvas = FigureCanvasTkAgg(fig, master=lf)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)


    def graficar_predicciones(real, prediccion):
        plt.plot(real[0:len(prediccion)],color='red', label='Valor real de la acción')
        plt.plot(prediccion, color='blue', label='Predicción de la acción')
        plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
        plt.xlabel('Tiempo')
        plt.ylabel('Valor de la acción')
        plt.legend()
        plt.show()

    def selectFile(self):
        fname = askopenfilename(filetypes=(("Archivo Dataset", "*.csv"),
                                           ("Hojas de calculo", "*.xlsx;*.xls"),
                                           ("Todos los archivos", "*.*") ))
        if fname:
            try:
                print(fname)
                self.fileSelect = fname
            except:
                print("")


    def entrenamiento(self):
        print("archivo a entrenar: "+self.fileSelect)
        dataset = pd.read_csv(self.fileSelect, index_col='Date', parse_dates=['Date'])
        dataset.head()
        set_entrenamiento = dataset[:'2016'].iloc[:,1:2]
        set_validacion = dataset['2017':].iloc[:,1:2]

        set_entrenamiento['High'].plot(legend=True)
        set_validacion['High'].plot(legend=True)
        plt.legend(['Entrenamiento (2006-2016)', 'Validación (2017)'])
        plt.show()

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

        graficar_predicciones(set_validacion.values,prediccion)




if __name__ == "__main__":
    root = tk.Tk()
    Application(root)#.pack(side="top", fill="both", expand=True)
    root.mainloop()