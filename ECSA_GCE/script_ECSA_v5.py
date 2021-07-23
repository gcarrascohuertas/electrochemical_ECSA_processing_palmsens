
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

np.warnings.filterwarnings('ignore')

SIZE_Font = 15

column_9 =  0 # Potential applied (V)
column_11 = 1 # WE(1).Current (A)
column_13 = 2 # Index


size=15
params = {'legend.fontsize': 'large',
          #'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)



#-------------------------------------PREAMBLE--------------------------------------------------
#load files
files = glob.glob('*.txt') #load every file with .txt extension 
print(files) #print in screen "files loaded"
#Raw text remarks
input_delimiter= "\t" #de raw data delimiter is tabular
input_skip_space=0 #we want 2 skip space from raw data
#Path in which we are working
path=os.getcwd() #get path 


label = ['5 mV/seg',"10 mV/seg",'20 mV/seg',"40 mV/seg","60 mV/seg","80 mV/seg","100 mV/seg","125 mV/seg","150 mV/seg","200 mV/seg"] #label set for graph
color = ['black','red','blue',"green","orange","purple","yellow","gray","olive","brown"] #colors set for graphs
scan_rate = [0.005,0.010,0.020,0.040,0.060,0.080,0.100,0.125,0.150,0.200]

#---------------------------------------------------------------------------------------
def new_directories(folder1,folder2):

    #This  function create new folders named as folder1 and folder2 in where will be stored figures and data separately and has been created 
    #by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact


    folder_1= folder1 #folder 1
    folder_2= folder2 #folder 2
    path= os.getcwd() + "\\" #obtain path
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

    path= os.getcwd() + "\\" #Obtain path 
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
def file_change_name(imput):

    #This  function change the name of O2 files in order to obtain ordered list in our directory and has been created by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact

    for name_old in imput: #for every file in our O2 list 

        a=name_old[5:-6] # characters located between 3 and -7 position
        b = float(a) # convert characters selected to float
        c = '%04i' % b # add 0 up to 4 positions before numbers converted to float 
        print("The file name with 0 placed is: " + c)
        name_new= name_old.replace(a, c)
        print("Renamed file is: " + name_new)
        print(name_old, ' ------- ',   name_new)
        os.rename(name_old,name_new)
    print("-------------------------------------END CHANGE NAME OF OXYGEN FILES FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def ECSA():

    #This  function plot all the scan rates performed in electrochemical surface area analysis  and has been created 
    #by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact

    files = glob.glob(path +'**/**/data_mean_CV*.txt',recursive=True) #load every file with .txt extension 
    print(files) #print in screen "files loaded"

    counter=0 #counter set to 0

    fig, ax = plt.subplots()

    for file in files:   ## For each file in files

        f = np.genfromtxt(file, delimiter="\t") #charge the file with tabular delimiter and skip header of 1 line
        WE_potential = f[:, 0] #Load from the "file" the column asociated to WE potential (V)
        WE_current = f[:, 1]  #Load from the "file" the column asociated to WE current (A)
        WE_current_corrected=WE_current #change WE current to microamperes

        

        ax.plot(WE_potential,WE_current_corrected,color=color[counter],label=label[counter]) #Plot the "WE potential (V)" vs  "WE current (A)"
        counter=counter+1 #we add 1 to counter for colors and labels

    #--------------------PLOTS OPTIONS--------------------------


    ax.tick_params(axis="y", right=True, direction='in')
    ax.tick_params(axis="x",top=True , direction='in')
    ax.legend(loc='lower right', prop={'size':12}) #graph legend

    ax.set_xlabel('Potencial vs. Ag/AgCl (V)',  fontsize= SIZE_Font)
    ax.set_ylabel('Intensidad de corriente (\u00B5A) ',  fontsize= SIZE_Font)

    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    ax.set_xlim(-0.1,1.2)
    ax.set_ylim(-100,100)  # Y axe limits
    #plt.grid()# paint a grid over the graph
    ax.figure.savefig("figure_ECSA_GCE_3mm.png")
    ax.figure.savefig("figure_ECSA_GCE_3mm.eps")



    shutil.move("figure_ECSA_GCE_3mm.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_ECSA_GCE_3mm.eps",  "figures") #Move the graph as "name.png" to folder figures
    #plt.show() # Show graph 
    #ax.clf() #clear the figure - you can still paint another plot onto it
    
    print("-------------------------------------END PLOT ECSA---------------------------------------------------")
#---------------------------------------------------------------------------------------
def operation():

    files_ox = glob.glob(path +'**/**/data_info_table_oxidation*.txt',recursive=True) #load every file with .txt extension 
    print(files_ox) #print in screen "files loaded"

    files_red = glob.glob(path +'**/**/data_info_table_reduction*.txt',recursive=True) #load every file with .txt extension 
    print(files_red) #print in screen "files loaded"
    list1=[]
    list2=[]


    lista_potencial_oxidacion = []
    lista_potencial_reduccion = []


    for file in files_ox:

        f1 = np.genfromtxt(file, delimiter=" ", skip_header=1)
        #print(f1)
        intersect_oxidation = f1[2]
        #print(intersect_oxidation)
        intensity_oxidation_peak = f1[1]
        potencial_oxidation_peak = f1[0]

        #print(intensity_oxidation_peak)
        ipa=intensity_oxidation_peak - intersect_oxidation 
        print("Intensity anodic peak current (microA):  " +str(ipa))
        list1.append(ipa)
        lista_potencial_oxidacion.append( potencial_oxidation_peak)

    for file in files_red:

        f2 = np.genfromtxt(file, delimiter=" ", skip_header=1)
        intersect_reduction = f2[2]
        intensity_reduction_peak = f2[1]
        potencial_reduction_peak = f2[0]

        ipc=intensity_reduction_peak - intersect_reduction

        print("Intensity cathodic peak current (microA):  " +str(ipc))
        list2.append(ipc)
        lista_potencial_reduccion.append(potencial_reduction_peak)

  
    array_1=np.array(list1)
    np.savetxt("data_ipa.txt", array_1.transpose())


    array_2=np.array(list2)
    np.savetxt("data_ipc.txt", array_2.transpose())


    array_3=np.array(lista_potencial_oxidacion)
    np.savetxt("data_lista_potencial_oxidacion.txt", array_3.transpose())

    
    array_4=np.array(lista_potencial_reduccion)
    np.savetxt("data_lista_potencial_reduccion.txt", array_4.transpose())


    print("-------------------------------------END OPERATION FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def Randles_Sevick_equation():

    """
    This  function calculate Area from Randle-Sevick equation acording to slope obtained of "ipa vs root square of scan rate" and has been created 
    by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact
    """

    n = 1   #number of electrons transferred in the redox event (usually 1)
    #A  == electrode area in cm2
    #F = 96485.33289  #Faraday Constant in C mol−1 == 96485.33289 mol−1
    D = 0.0000073  #diffusion coefficient in cm2/s == A 25°C los valores Do  (coeficiente de difusión) para el K_3 [Fe(CN)_6] en KCl 0,10 M son 7.3∗〖10〗^(−6) cm2 / s, respectivamente
    C = 0.000004  #concentration in mol/cm3 ----0.004 mol/L
    R = 8.3144598   #Gas constant in J K−1 mol−1  ==8.3144598 J⋅mol−1⋅K−1
    T = 298  #temperature in K = 25ºC ==298 K
    #Randles–Sevcik equation (anodic peak)
    #ip=268600*(n**3/2)*(A)*(D**0.5)*(C)*(value**0.5)

    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    list_1 = []
    list_2 = []

    #--------------------------------------CATHODIC PROCESS-------------------------------------

    f_1 = np.genfromtxt("data_ipc.txt")
    ipc = np.array(f_1)
    #print(ipc)
    ip_1 = ipc  # Cathodic peak current in microamps (microA)
    ip_corrected_1=ip_1*0.000001 #Cathodic  peak current  in amps (A) 

    list_1=(ipc,ip_corrected_1) #list with anodic peak current in amps and in microamps
    array_1=np.array(list_1) #array of list 
    #print(array_1)
    tabla_1= Table(array_1.transpose()) #table of array
    new_column_1=Column([0.005,0.010,0.020,0.040,0.060,0.080,0.100,0.125,0.150,0.200], name="Scan rate (V/seg)")
    tabla_1.add_column(new_column_1) #add new column 
    tabla_1.write("data_info_table_oxidation_global.txt", format='ascii')

    f_2 = np.genfromtxt("data_info_table_oxidation_global.txt", delimiter=" ", skip_header=1)
    scan_rate_list_1 = f_2[:, 2]  #scan rate in V/s
    print(scan_rate_list_1)

    #-------------------- ip vs v^1/2 LINEAR REGRESSION--------------------------
    #---------FIRST-----------
    w_1=0 #start value of range
    w_2=0.5 #end value of range
    i_interval_1 = np.where( ((scan_rate_list_1**0.5) < w_2) & ((scan_rate_list_1**0.5) > w_1) )[0] #range of regresion
    x_interval_1 = (scan_rate_list_1**0.5)[i_interval_1] #find in X axis the range of regresion
    y_interval_1 = ip_corrected_1[i_interval_1] #find in Y axis the range of regresion

    #---------LINEAR REGRESSION-----------
    adjust_1 = np.polyfit(x_interval_1, y_interval_1, deg=1)
    y_adjust_1 = np.polyval(adjust_1, x_interval_1)
    a_1 = adjust_1[0] #slope
    b_1 = adjust_1[1] #independ variable
    print("x-value and y-value fits the linear regression as A and B= "+  str(adjust_1))

    def lsqfity(X, Y):

        """
        Calculate a "MODEL-1" least squares fit.

        The line is fit by MINIMIZING the residuals in Y only.

        The equation of the line is:     Y = my * X + by.

        Equations are from Bevington & Robinson (1992)
        Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
        pp: 104, 108-109, 199.

        Data are input and output as follows:

        my, by, ry, smy, sby = lsqfity(X,Y)
        X     =    x data (vector)
        Y     =    y data (vector)
        my    =    slope
        by    =    y-intercept
        ry    =    correlation coefficient
        smy   =    standard deviation of the slope
        sby   =    standard deviation of the y-intercept

        """

        X, Y = map(np.asanyarray, (X, Y))

        # Determine the size of the vector.
        n = len(X)

        # Calculate the sums.

        Sx = np.sum(X)
        Sy = np.sum(Y)
        Sx2 = np.sum(X ** 2)
        Sxy = np.sum(X * Y)
        Sy2 = np.sum(Y ** 2)

        # Calculate re-used expressions.
        num = n * Sxy - Sx * Sy
        den = n * Sx2 - Sx ** 2

        # Calculate my, by, ry, s2, smy and sby.
        my = num / den
        by = (Sx2 * Sy - Sx * Sxy) / den
        ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

        diff = Y - by - my * X

        s2 = np.sum(diff * diff) / (n - 2)
        smy = np.sqrt(n * s2 / den)
        sby = np.sqrt(Sx2 * s2 / den)



        return my, by, ry, smy, sby    



    print(lsqfity(x_interval_1,y_interval_1))






    #-------------------- AREA CALCULATING--------------------------
    Area_final_cathodic=(a_1)/((269000)*(1)*(C)*((D**0.5))) #calculate the area
    print("Area (cm2) Anodic " + str(Area_final_cathodic))



    text_file_1 = open("data_area_cathodic_convolution.txt", "w")
    n_1 = text_file_1.write(str(Area_final_cathodic))
    text_file_1.close()


    #--------------------------------------ANODIC PROCESS-------------------------------------

    f_3 = np.genfromtxt("data_ipa.txt")
    ipa = np.array(f_3)
    #print(ipa)
    ip_2 = ipa  # Cathodic peak current in microamps (microA)
    ip_corrected_2=ip_2*0.000001 #Cathodic  peak current  in amps (A) 

    list_2=(ipa,ip_corrected_2) #list with anodic peak current in amps and in microamps
    array_2=np.array(list_2) #array of list 
    #print(array_2)
    tabla_2= Table(array_2.transpose()) #table of array
    new_column_2=Column([0.005,0.010,0.020,0.040,0.060,0.080,0.100,0.125,0.150,0.200], name="Scan rate (V/seg)")
    tabla_2.add_column(new_column_2) #add new column 
    tabla_2.write("data_info_table_reduction_global.txt", format='ascii')

    f_4 = np.genfromtxt("data_info_table_reduction_global.txt", delimiter=" ", skip_header=1)
    scan_rate_list_2 = f_4[:, 2]  #scan rate in V/s
    print(scan_rate_list_2)

    #-------------------- ip vs v^1/2 LINEAR REGRESSION--------------------------
    #---------FIRST-----------
    w_1_2=0 #start value of range
    w_2_2=0.5 #end value of range
    i_interval_2 = np.where( ((scan_rate_list_2**0.5) < w_2_2) & ((scan_rate_list_2**0.5) > w_1_2) )[0] #range of regresion
    x_interval_2 = (scan_rate_list_2**0.5)[i_interval_2] #find in X axis the range of regresion
    y_interval_2 = ip_corrected_2[i_interval_2] #find in Y axis the range of regresion

    #---------LINEAR REGRESSION-----------
    adjust_2 = np.polyfit(x_interval_2, y_interval_2, deg=1)
    y_adjust_2 = np.polyval(adjust_2, x_interval_2)
    a_2 = adjust_2[0] #slope
    b_2 = adjust_2[1] #independ variable
    print("x-value and y-value fits the linear regression as A and B= "+  str(adjust_2))

    def lsqfity(X, Y):

        """
        Calculate a "MODEL-1" least squares fit.

        The line is fit by MINIMIZING the residuals in Y only.

        The equation of the line is:     Y = my * X + by.

        Equations are from Bevington & Robinson (1992)
        Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
        pp: 104, 108-109, 199.

        Data are input and output as follows:

        my, by, ry, smy, sby = lsqfity(X,Y)
        X     =    x data (vector)
        Y     =    y data (vector)
        my    =    slope
        by    =    y-intercept
        ry    =    correlation coefficient
        smy   =    standard deviation of the slope
        sby   =    standard deviation of the y-intercept

        """

        X, Y = map(np.asanyarray, (X, Y))

        # Determine the size of the vector.
        n = len(X)

        # Calculate the sums.

        Sx = np.sum(X)
        Sy = np.sum(Y)
        Sx2 = np.sum(X ** 2)
        Sxy = np.sum(X * Y)
        Sy2 = np.sum(Y ** 2)

        # Calculate re-used expressions.
        num = n * Sxy - Sx * Sy
        den = n * Sx2 - Sx ** 2

        # Calculate my, by, ry, s2, smy and sby.
        my = num / den
        by = (Sx2 * Sy - Sx * Sxy) / den
        ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

        diff = Y - by - my * X

        s2 = np.sum(diff * diff) / (n - 2)
        smy = np.sqrt(n * s2 / den)
        sby = np.sqrt(Sx2 * s2 / den)



        return my, by, ry, smy, sby    



    print(lsqfity(x_interval_2,y_interval_2))





    #-------------------- AREA CALCULATING--------------------------
    Area_final_anodic=abs((a_2)/((269000)*(1)*(C)*((D**0.5)))) #calculate the area
    print("Area (cm2) Cathodic " + str(Area_final_anodic))

    text_file_2 = open("data_area_anodic_convolution.txt", "w")
    n_2 = text_file_2.write(str(Area_final_anodic))
    text_file_2.close()


    #--------------------PLOTS--------------------------


    ax1.plot((scan_rate_list_1**0.5),ip_corrected_1,"o", color='black') #"plot root square of scan rate" vs "ipc"
    ax2.plot((scan_rate_list_2**0.5),ip_corrected_2,"o", color='red') #"plot root square of scan rate" vs "ipc"


    #--------------------PLOTS OPTIONS--------------------------
    #plt.title("Randles Sevick plot area ") # graph title
    #plt.ylim(-50,50)   # Y axe limits
    #plt.grid() # paint a grid over the graph
    #plt.show() # Show graph 





    ax1.tick_params(axis="y", right=True, direction='in', labelcolor="red")
    ax1.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='lower right', prop={'size':8}) #graph legend
    ax1.set_xlabel(r"Velocidad de barrido $v^\frac{1}{2}$ (V/s) ",  fontsize= SIZE_Font)
    ax1.set_ylabel('Ip anódica (A)',  fontsize= SIZE_Font,  color="red")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    ax1.set_xlim(0,0.5)
    #plt.grid()# paint a grid over the graph
    # Change the y ticklabel format to scientific format
    ax1.set_ylim(-0.0001,0.0001)  # Y axe limits
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))



    ax2.tick_params(axis="y", right=True, direction='in',  labelcolor="black")
    ax2.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='lower right', prop={'size':8}) #graph legend
    ax2.set_xlabel(r"Velocidad de barrido $v^\frac{1}{2}$ (V/s) ",  fontsize= SIZE_Font)
    ax2.set_ylabel('Ip catódica (A)',  fontsize= SIZE_Font,  color="black")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    ax2.set_xlim(0,0.5)
    ax2.set_ylim(-0.0001,0.0001)  # Y axe limits
    #plt.grid()# paint a grid over the graph
    # Change the y ticklabel format to scientific format
    #ax1.set_ylim(0,0.0005)  # Y axe limits
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped


    ax1.figure.savefig("figure_randles_sevick_plot_ECSA_GCE_3mm.png",bbox_inches='tight')
    ax1.figure.savefig("figure_randles_sevick_plot_ECSA_GCE_3mm.eps",bbox_inches='tight')
    shutil.move("figure_randles_sevick_plot_ECSA_GCE_3mm.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_randles_sevick_plot_ECSA_GCE_3mm.eps",  "figures") #Move the graph as "name.png" to folder figures




    plt.clf() #clear the figure - you can still paint another plot onto it
    

    print("-------------------------------------END RANDLES-SEVICK FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def Deltap():


    f_1 = np.genfromtxt("data_lista_potencial_oxidacion.txt")
    ppc = np.array(f_1)
    #print(ppc)
 
    f_2 = np.genfromtxt("data_lista_potencial_reduccion.txt")
    ppa = np.array(f_2)
    #print(ppa)





    potencial_oxidacion= f_1[0]


    #print(potencial_oxidacion)

    potencial_reduccion = f_2[0]
    #print(potencial_reduccion)

    resta=  ppc -  ppa
    print(resta)



def move_data():

    """ This function move every data file .txt to folder data and
    has been created by Gaspar, gasparcarrascohuertas@gmail.com for contact
    """

    files = glob.glob(path+ "/data*.txt")
    print(files)
    print("Files in moved to directories are: "+str(files))
    for f in files:
     shutil.move(f, "data")
    print("-------------------------------------END MOVING DATA .TXT  FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
new_directories("figures", "data")
removing_files_in_folder("figures","data")
#file_change_name(files)
ECSA()
operation()
Randles_Sevick_equation()
Deltap()
move_data()

exit()








