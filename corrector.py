from datetime import timedelta, date
import pandas as pd
import numpy as np
import csv

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

import csv

with open('data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

datos = []
for i in range(1,len(data)):
    datos.append(str(data[i][0]))

start_date = date(2018, 1, 1)
end_date = date(2020, 9, 4)
cont = 0
cati=""
for single_date in daterange(start_date, end_date):
    cati+=str(single_date.strftime("%Y-%m-%d"))
    cati+=str(",")
    if str(single_date.strftime("%Y-%m-%d")) in datos:
        cont+=1
        cati+=str(data[cont][1])
    else:
        cati+=str("0")
    #cati+=","+str(single_date.weekday())+","
    #cati+=str(single_date.strftime("%m"))
    cati+="\n"
f= open("datafinal.csv","w+")
print(cati)
f.write(cati)
f.close()
