# -*- coding: utf-8 -*-

import matplotlib
import io
import astropy
import csv
import sys
from astropy.table import Table, Column, MaskedColumn
from astropy.stats import SigmaClip
from numpy import mean
from numpy import std
from scipy.integrate import simps
from astropy.convolution import Gaussian1DKernel, convolve
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.misc import derivative
from scipy import stats
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
import sympy as sp
from astropy.table import Table, Column
from scipy.optimize import fsolve
from matplotlib import pylab
import glob #module wich create list of strings of a directory
import shutil #module which moves files
import os 
from astropy.io import ascii
from numpy.linalg import inv
import matplotlib.pyplot as plt

from numpy.linalg import inv

import scipy as scipy
from scipy import optimize
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker


path=os.getcwd() #get path of directory
concentracion=float(input("Introduzca la concentracion: "))
label_list = ['1 rep',"2 rep",'3 rep','4 rep']
color_list= ['black','red','blue','green'] #colos set for graphs
#---------------------------------------------------------------------------------------
def new_directories(folder1,folder2):

    #This  function create new folders named as folder1 and folder2 in where will be stored figures and data separately and has been created 
    #by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact


    folder_1= folder1 #folder 1
    folder_2= folder2 #folder 2
    path= os.getcwd() + "/" #obtain path
    dir1 = path+folder_1  #'path_to_my_folder_1'
    dir2 = path+folder_2  #'path_to_my_folder_2'

    print(dir1)
    print(dir2)

    if not os.path.exists(dir1 and dir2): # if the directory does not exist
        os.makedirs(dir1 ) # make the directory 1
        os.makedirs(dir2)  # make the directory 2
    else: # the directory exists
        #removes all files in a folder
        for the_file in os.listdir(dir1 and dir2):
            file_path = os.path.join(dir1, the_file)
            file_path = os.path.join(dir2, the_file)
    print("-------------------------------------END CREATE NEW DIRECTORIES FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def removing_files_in_folder(folder1,folder2):

    #This  function remove every file in new folders named as folder1 and folder2  and has been created 
    #by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact

    path= os.getcwd() + "/" #Obtain path 
    folder_1= folder1 #folder 1
    folder_2= folder2 #folder 2

    list_files = os. listdir(path+folder1)  #list all the files in folder 1

    files_figures = glob.glob(path+folder1+ "/*") 
    for f in files_figures:  #for every file  file in folder 2 remove
     os.remove(f)  

    files_data = glob.glob(path+folder2+ "/*")
    for f in files_data:   #for every file  file in folder 2 remove
     os.remove(f) 

    print("-------------------------------------END REMOVING FILES IN FOLDERS FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def tratamiento_csv_rawdata():
    #This  function create new stack graphs, create new .csv files from original ones
    #and create new file with mean and std. dev 
    #and has been created by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact

    path = os.getcwd()
    extension = 'csv'
    os.chdir(path)
    archivos = glob.glob('*.{}'.format(extension))
    print("Los archivos cargados son : " + str(archivos))
    ax = plt.gca()
    df_list = []
    n=0

    for archivo in archivos:

        df = pd.read_csv(archivo, encoding='utf-16') #generate dataframe of pandas from csv
        df.rename(columns={df.columns[0]: 'Voltage', df.columns[1]: 'Intensity'}, inplace=True) #rename headings of columns 0 and 1
        df.drop(df.index[:5], inplace=True) #removes first 5 rows
        df=df[:806].astype(float) # converting to float

        x=df['Voltage']
        y=df['Intensity'] 
        df.to_csv(r"CV_generated_file_" +str(n)+".txt", header=None, index=None, sep=' ', mode='a') # if you want to store as .txt
        print("Los archivos de salida son son : " + "CV_generated_file_" +str(n)+".txt")

        a=list(df['Voltage'].array) 
        b=list(df['Intensity'].array)
        c=np.array(list(zip(a,b)))
        df_list.append(b)

        
        n= n +1  


    new_array = np.array(df_list)
    media = np.mean(new_array, axis=0)
    desviacion_std=np.std(new_array, axis=0)

    print("-------------------------------------END PLOT DATA.CSV  FUNCTION---------------------------------------------------")
#-------------------------------------------------------------------------------------
def new_generador(color1,label1):
    archivos = glob.glob('CV_generated_file_*.txt')[:3]
    print("Los archivos cargados son : " + str(archivos))
    datos=[]
    lista = []
    xcol=0
    ycol=1
    n=0
    lista_potential= []

    Tabla=Table()
    fig, ax = plt.subplots()

  

    for archivo in archivos[:1]:
        f=np.genfromtxt(archivo, delimiter=" ", skip_header=2)
        x = f[:, xcol]
        lista_potential.append(x)
        #print(lista_potential)
        array_potential = np.array(lista_potential)
        #column_1 = Column(array_potential, name='WE_potential(V)')
        #Tabla.add_column(column_1, index=0)
        #np.savetxt("data_potential_applied_" + str(concentracion) +".txt", array_potential) 

    #print("Esto es la tabla")
    #print(Tabla)

    for archivo in archivos:
        f=np.genfromtxt(archivo, delimiter=" ", skip_header=2)
        x = f[:, xcol]
        y = f[:, ycol]

        lista.append(f)

        
        #----------------------------------------------
        #Restrigimos con un intervalo de x con valor dado para hacer el máximo
        x1=(0) #valor de inicio del intervalo
        x2=(1) #valor de fin del intervalo
        i_interval = np.where( (x < x2) & (x > x1) )[0] #restrigimos con un intervalo de x con valor dado
        x_interval = x[i_interval]#valor del intervalo
        y_interval = y[i_interval]#valor del intervalo
        #Hacemos el mínimo
        minimo_y = np.min( y_interval )
        indice1 = np.where(y_interval == minimo_y)[0]  ## te devuelve un array con las posiciones donde y es igual a ymin
        minimo_x = x_interval[indice1]   ## xmin es el valor de x al que corresponde el ymin
        print("Minimum of  function in x is:", minimo_x, "also y:", minimo_y)
        #----------------------------------------------
        #Y-value correction straight baseline
        ax.plot(x, y,'-',color=color1[n],label=label1[n])
        datos.append(minimo_y)

        n=n+1

    #-----------PLOT FIGURE-------------

    ax.tick_params(axis="y", right=True, direction='in')
    ax.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='upper left', prop={'size':12}) #graph legend
    ax.set_xlabel("Potencial vs. Ag/AgCl, 3.5 M (V) ")
    ax.set_ylabel('Intensidad de corriente (\u00B5A)')
    ax.set_xlim(-0.1,0.7)
    ax.set_ylim(-200,200)  # Y axe limits
    ax.figure.savefig("figure_1.eps", format="eps")
    ax.figure.savefig("figure_1.png", format="png")

    plt.clf()
    #------------------------

    new_array = np.array(lista)
   
    media = np.mean(new_array, axis=0)
    desviacion_std=np.std(new_array, axis=0)
    np.savetxt("data_mean_CV_" + str(concentracion) +".txt", media) 


    np.savetxt("data_dev_std_"+ str(concentracion) +".txt", desviacion_std) 
    print("-------------------------------------END NEW GENERADOR  FUNCTION---------------------------------------------------")
#-------------------------------------------------------------------------------------
def plot_mean_std_dev():
    
    xcol = 0
    ycol = 1

    fig, ax = plt.subplots()
    fig, ax2 = plt.subplots()


    text1 = np.genfromtxt("data_mean_CV_"+str(concentracion)+".txt", delimiter='')
    x = text1[:, xcol]
    y = text1[:, ycol]
    y_correct_baseline=y
    electrode_area=0.0706 #cm2
    y_correct_baseline_density=y/electrode_area
    text2 = np.genfromtxt("data_dev_std_"+str(concentracion)+".txt", delimiter='')
    y2 = text2[:, ycol]


    ax.errorbar(x, y_correct_baseline, yerr=y2,color='red')
    ax.plot(x, y_correct_baseline,'-',color='red',label='1',linewidth=1)

    ax.tick_params(axis="y", right=True, direction='in')
    ax.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='upper left', prop={'size':12}) #graph legend
    ax.set_xlabel("Potencial vs. Ag/AgCl ( V) ")
    ax.set_ylabel('Intensidad de corriente (\u00B5A)')
    ax.set_xlim(-0.1,0.7)
    ax.set_ylim(-200,200)  # Y axe limits
    ax.figure.savefig("figure_2_mean.eps", format="eps")
    ax.figure.savefig("figure_2_mean.png", format="png")



    ax2.errorbar(x, y_correct_baseline_density, yerr=y2,color='red')
    ax2.plot(x, y_correct_baseline_density,'-',color='red',label='1',linewidth=1)


    ax2.tick_params(axis="y", right=True, direction='in')
    ax2.tick_params(axis="x",top=True , direction='in')
    ax2.set_xlabel("Potencial vs. Ag/AgCl, 3.5 M (V) ")
    ax2.set_ylabel('Densidad de corriente (\u00B5A/cm2)')
    ax2.set_xlim(-0.1,0.7)
    ax2.set_ylim(-2000,2000)  # Y axe limits
    ax2.figure.savefig("figure_3_mean_current_density.eps", format="eps")
    ax2.figure.savefig("figure_3_mean_current_density.png", format="png")
    plt.clf()
#---------------------------------------------------------------------------------------
def Add_index_data():

    counter=0
    file = "data_mean_CV_"+str(concentracion)+".txt"
    df = pd.read_csv(file, sep=" ")
    
    df.rename(index={0:'Index'})
    # changing columns using .columns() 
    df.columns = ['WE Potential ', 'WE Current ']
    # creating a list of dataframe columns 
    #print(df)
    tabla = Table.from_pandas(df)
    #print(tabla)

    
    fname = "data_mean_CV_"+str(concentracion)+".txt"
    num_lines = -1

    with open(fname, 'r') as f:
        for line in f:
            num_lines += 1
    print("Number of lines:")
    #print(num_lines)

    l = [i for i in range(num_lines)]
    new_column=Column(l, name="Index")
    tabla.add_column(new_column)
    #print(tabla)
    tabla.write(fname, format='ascii',delimiter="\t", overwrite=True)
    


    print("---------------------------------------------------ADD INDEX TO  DATA ---------------------------------------------------")
#---------------------------------------------------------------------------------------
def ECSA_reduction_branch():

    #This  function plot all the reduction branch of scan rates performed in electrochemical surface area analysis  and has been created 
    #by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact

    counter=0 #counter set to 0

    data_list_1=[] #empty data list
    data_list_2=[] #empty data list
    data_list_3=[] #empty data list
    data_list_4=[] #empty data list
    fig, ax = plt.subplots()

    file = "data_mean_CV_"+str(concentracion)+".txt"
    f = np.genfromtxt(file, delimiter="\t", skip_header=1) #charge the file with tabular delimiter and skip header of 1 line
    WE_potential = f[:802, 0] #Load from the file the column asociated to WE potential (V)
    WE_current = f[:802, 1] #Load from the file the column asociated to WE current (A)
    WE_current_corrected=WE_current #change WE current to microamperes
    index=  f[:802, 2] #Load from the file the column asociated to index (a.u.)


    #--------------------LINEAR REGRESSION--------------------------
    #we need to define two analysis ranges for the regression
    #---------FIRST-----------
    time1= 450 #Start interval
    time2= 455 #End interval
    i_interval_1 = np.where( (index< time2) & (index > time1) )[0] # defining interval
    x_interval_1 = WE_potential[i_interval_1] #Find in my interval values asociated to x (WE potential (V))
    y_interval_1 = WE_current_corrected[i_interval_1]#Find in my interval values asociated to y (WE current (microA))
    #---------SECOND-----------
    tiempo3= 460 #Start interval
    tiempo4= 475 #End interval
    i_interval_2 = np.where( (index < tiempo4) & (index > tiempo3) )[0] # defining interval
    x_interval_2 = WE_potential[i_interval_2]#Find in my interval values asociated to x (WE potential (V))
    y_interval_2 = WE_current_corrected[i_interval_2]#Find in my interval values asociated to y (WE current (microA))
    
    print(x_interval_1)
    print(x_interval_2)
    
    
    #---------LINEAR REGRESSION-----------
    adjust = np.polyfit(x_interval_2, y_interval_2, deg=1) # linealice the x-values and y-values of the interval to polynomial function order 1
    y_adjust = np.polyval(adjust, x_interval_1) #applying the polynomial function which you got using polyfit
    print("x-value and y-value fits the linear regression as A and B= "+  str(adjust))
    #print(file)
    data_list_1.append(adjust)


    #--------------------FIND LOCAL MINIMUM--------------------------
    #we need to define analysis range in which minimum is located 
    x1_1=(-0.1) #Start interval
    x1_2=(0.4) #End interval

    i_interval_min = np.where( (WE_potential < x1_2) & (WE_potential > x1_1) )[0] # defining interval
    x_interval_min = WE_potential[i_interval_min] #Find in my interval x- values asociated to WE potential (V)
    y_interval_min = WE_current_corrected[i_interval_min] #Find in my interval y-values asociated to WE current (microA)
    min_y = np.min( y_interval_min )
    index1 = np.where(y_interval_min == min_y)[0]  ## Array in which positions y = ymin
    min_x = x_interval_min[index1][0]   ## xmin es el valor de x al que corresponde el ymin
    text_min = (str(min_x)+ ";"+ str(min_y)+ "\n")
    print("Minimum x-value of function is:", min_x, "with y-value:", min_y)
    data_list_2.append(min_x) 
    data_list_3.append(min_y) 
  
    #--------------------FIND INTERSECTION MINIMUM--------------------------
    #we need to know the intersection between the linear regression and the x-minimum  value
    y_intersec = np.polyval(adjust, min_x)
    print("y-value in intersection is " + str(y_intersec))
    print("----------------------------------------")
    data_list_4.append(y_intersec) 
    
    #--------------------PLOTS--------------------------
    ax.vlines(min_x, -1000, 1000,color='k', linestyle='--') # plot vertical lines in x-value for minimum from -100 to 100 color black
    ax.plot(min_x, y_intersec, 'Xg') #plot green point in the x-value for minimum and y-value for intersection 
    ax.plot(x_interval_1, y_adjust,'k',linestyle='--') #plot 
    #ax.plot(x_interval_1, y_interval_1,'-',color='k',label='1',linewidth=1)
    ax.plot(WE_potential,WE_current_corrected,color="black") #plot x-values for potential(V) and y-values for current(microA) 
    

    #--------------------TABLES--------------------------
    data = (data_list_2,data_list_3,data_list_4) 
    data_array=np.array(data)
    tabla= Table(data_array.transpose())
    new_column=Column([concentracion/1000], name="Scan rate (V/seg)")
    tabla.add_column(new_column)
    tabla.rename_column("col0","WE Potential min (V)") # Rename column 0 to  max_x (V)
    tabla.rename_column("col1","WE Current min (\u00B5A)") # Rename column 1 to  max_y (microA)
    tabla.rename_column("col2","y-axe intersection (\u00B5A)") # Rename column 2 to  y_intersec (microA)
    print(tabla)
    tabla.write("data_info_table_reduction_" + str(concentracion)+ ".txt", format='ascii')
   
    
    #--------------------PLOTS OPTIONS--------------------------
    ax.tick_params(axis="y", right=True, direction='in')
    ax.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='upper left', prop={'size':12}) #graph legend
    ax.set_xlabel("Potencial vs. Ag/AgCl, 3.5 M (V) ")
    ax.set_ylabel('Intensidad de corriente (\u00B5A)')
    ax.set_xlim(-0.1,0.7)
    ax.set_ylim(-200,200)  # Y axe limits
    ax.figure.savefig("figure_reduction.eps", format="eps")
    ax.figure.savefig("figure_reduction.png", format="png")
    shutil.move("figure_reduction.png",  "figures")#Move the graph as "name.png" to folder figures
    shutil.move("figure_reduction.eps",  "figures")#Move the graph as "name.png" to folder figures
    plt.clf()



    
    print("---------------------------------------------------END ECSA REDUCION BRANCH FUNCTION---------------------------------------------------")

def ECSA_oxidation_branch():

    #This  function plot all the oxidation branch of scan rates performed in electrochemical surface area analysis  and has been created 
    #by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact
    
    counter=0 #counter set to 0

    data_list_1=[] #empty data list
    data_list_2=[] #empty data list
    data_list_3=[] #empty data list
    data_list_4=[] #empty data list

    t=Table() #creates new empty table
    fig, ax = plt.subplots()


    file = "data_mean_CV_"+str(concentracion)+".txt"
    f = np.genfromtxt(file, delimiter="\t", skip_header=1) #charge the file with tabular delimiter and skip header of 1 line
    WE_potential = f[:802, 0] #Load from the file the column asociated to WE potential (V)
    WE_current = f[0:802, 1] #Load from the file the column asociated to WE current (A)
    WE_current_corrected=WE_current #change WE current to microamperes
    index=  f[0:802, 2] #Load from the file the column asociated to index (a.u.)




        
    #--------------------LINEAR REGRESSION--------------------------
    #we need to define two analysis ranges for the regression
    #---------FIRST-----------
    time1= 100 #Start interval
    time2= 105 #End interval
    i_interval_1 = np.where( (index < time2) & (index > time1) )[0] # defining interval
    x_interval_1 = WE_potential[i_interval_1] #Find in my interval values asociated to x (WE potential (V))
    y_interval_1 = WE_current_corrected[i_interval_1] #Find in my interval values asociated to y (WE current (microA))
    #---------SECOND-----------
    #intervalo para regresion
    tiempo3= 110  #Start interval
    tiempo4= 120  #End interval
    i_interval_2 = np.where( (index < tiempo4) & (index > tiempo3) )[0] # defining interval
    x_interval_2 = WE_potential[i_interval_2] #Find in my interval values asociated to x (WE potential (V))
    y_interval_2 = WE_current_corrected[i_interval_2] #Find in my interval values asociated to y (WE current (microA))
    #---------LINEAR REGRESSION-----------
    adjust = np.polyfit(x_interval_2, y_interval_2, deg=1) # linealice the x-values and y-values of the interval to polynomial function order 1
    y_adjust = np.polyval(adjust, x_interval_1) #applying the polynomial function which you got using polyfit
    print(file)
    #print("x-value and y-value fits the linear regression as A and B= "+  str(adjust))
    data_list_1.append(adjust)
        
    #--------------------FIND LOCAL MAXIMUM--------------------------
    #we need to define analysis range in which minimum is located 
    x2_1=(0.0) #Start interval
    x2_2=(0.5) #End interval
    i_interval = np.where( (WE_potential < x2_2) & (WE_potential > x2_1) )[0] # defining interval
    x_interval = WE_potential[i_interval] #Find in my interval values asociated to x (WE potential (V))
    y_interval = WE_current_corrected[i_interval] #Find in my interval values asociated to y (WE current (microA))
    max_y = np.max( y_interval )
    index2 = np.where(y_interval == max_y)[0]  ## Array in which positions y = ymax
    max_x = x_interval[index2][0]  ## xmax es el valor de x al que corresponde el ymax
    #Files with data extracted from x-y-max  .txt
    text_max = (str(max_x)+ ";"+ str(max_y)+ "\n")        
    print("Maximum x-value of function is :", max_x, "with y-value:", max_y)
    data_list_2.append(max_x)   
    data_list_3.append(max_y)      
        
    #--------------------FIND INTERSECTION MAXIMUM--------------------------
    #we need to know the intersection between the linear regression and the x-maximum  value
    y_intersec = np.polyval(adjust, max_x)
    print("y-value in intersection is: " + str(y_intersec))
    print("----------------------------------------")
    data_list_4.append(y_intersec) 

    #--------------------PLOTS--------------------------
    ax.vlines(max_x, -10000, 10000,color='k', linestyle='--') # plot vertical lines in x-value for minimum from -100 to 100 color black
    ax.plot(max_x, y_intersec, 'Xg')  #plot green point in the x-value for minimum and y-value for intersection 
    ax.plot(x_interval_1, y_adjust,'k',linestyle='--') #plot 
    ax.plot(x_interval_1, y_interval_1,'-',color='k',label='1',linewidth=1)
    ax.plot(WE_potential,WE_current_corrected,color='red') #plot x-values for potential(V) and y-values for current(microA) 
        
    counter=counter+1
    
    #--------------------TABLES--------------------------
    data = (data_list_2,data_list_3,data_list_4) #max_x , max_y , y_intersec
    data_array=np.array(data)
    tabla= Table(data_array.transpose())
    new_column=Column([concentracion/1000], name="Scan rate (V/seg)")
    tabla.add_column(new_column)
    tabla.rename_column("col0","WE Potential max (V)") # Rename column 0 to  max_x (V)
    tabla.rename_column("col1","WE Current max (\u00B5A)") # Rename column 1 to  max_y (microA)
    tabla.rename_column("col2","y-axe intersection (\u00B5A)") # Rename column 2 to  y_intersec (microA)
    print(tabla)
    tabla.write("data_info_table_oxidation_" + str(concentracion)+ ".txt", format='ascii')
    
    
    #--------------------PLOTS OPTIONS--------------------------

    ax.tick_params(axis="y", right=True, direction='in')
    ax.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='upper left', prop={'size':12}) #graph legend
    ax.set_xlabel("Potencial vs. Ag/AgCl, 3.5 M (V) ")
    ax.set_ylabel('Intensidad de corriente (\u00B5A)')
    ax.set_xlim(-0.1,0.7)
    ax.set_ylim(-200,200)  # Y axe limits
    ax.figure.savefig("figure_oxidation.eps", format="eps")
    ax.figure.savefig("figure_oxidation.png", format="png")
    shutil.move("figure_oxidation.png",  "figures")#Move the graph as "name.png" to folder figures
    shutil.move("figure_oxidation.eps",  "figures")#Move the graph as "name.png" to folder figures
    plt.clf()


    print("-------------------------------------END ECSA OXIDATION BRANCH FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------

def Deltap():


    text1 = np.genfromtxt("data_info_table_oxidation_" + str(concentracion)+ ".txt", delimiter="\t")
    potencial_oxidation = text1[0]
    print(potencial_oxidation)


    text2 = np.genfromtxt("data_info_table_reduction_"+str(concentracion)+".txt", delimiter="\t")
    potencial_reduccion = text2[0]

    potencial_resta= potencial_oxidation - potencial_reduccion

    print(potencial_oxidation)
    

    print("-------------------------------------END Delta P FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------


def move_data():

    """ This function move every data file .txt to folder data and
    has been created by Gaspar, gasparcarrascohuertas@gmail.com for contact
    """

    #files1 = glob.glob(path+ "/CV_generated_file_*.csv")
    files2 = glob.glob(path+ "/CV_generated_file_*.txt")
    files3 = glob.glob(path+ "/data_*.txt")


    #print("Files  moved to directories are: "+str(files1))
    print("Files  moved to directories are: "+str(files2))
    print("Files  moved to directories are: "+str(files3))

    #for f in files1:
     #shutil.move(f, "data")
    for f in files2:
     shutil.move(f, "data")
    for f in files3:
     shutil.move(f, "data")


    graphs1 = glob.glob(path+ "/figure*.png")
    print("Files  moved to directories are: "+str(graphs1))
    graphs2 = glob.glob(path+ "/figure*.eps")
    print("Files moved to directories are: "+str(graphs2))
    for g in graphs1:
     shutil.move(g, "figures")
    for g in graphs2:
     shutil.move(g, "figures")
    print("-------------------------------------END MOVING DATA.CSV and FIGURES.eps .png FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
new_directories("figures", "data")
removing_files_in_folder("figures","data")
tratamiento_csv_rawdata()
new_generador(color_list,label_list)
plot_mean_std_dev()
Add_index_data()
ECSA_reduction_branch()
ECSA_oxidation_branch()
Deltap()
move_data()
exit()


