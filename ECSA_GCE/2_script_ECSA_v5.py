# -*- coding: utf-8 -*-

# Copyright (C) 2021 Gaspar Carrasco-Huertas
# gasparcarrascohuertas@gmail.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import matplotlib # import matplotlib library
import io # import io library
import astropy # import astropy library
import csv # import csv library
import sys # import sys library
from astropy.table import Table, Column, MaskedColumn # import astropy library
from astropy.stats import SigmaClip # import astropy library
from numpy import mean  # import numpy library
from numpy import std # import numpy library
from scipy.integrate import simps # import scipy library
from astropy.convolution import Gaussian1DKernel, convolve # import astropy library
import numpy as np  # import numpy library
import matplotlib.pyplot as plt  # import matplotlib library
import scipy # importscipy library
from scipy.optimize import curve_fit # import scipy library
from scipy.misc import derivative # import scipy library
from scipy import stats # import scipy library
import pandas as pd # import pandas library
import uncertainties as unc # import uncertainties library
import uncertainties.unumpy as unp # import uncertainties library
import sympy as sp # import simpy library
from astropy.table import Table, Column # import astropy library
from scipy.optimize import fsolve # import scipy library
from matplotlib import pylab  # import matplotlib library
import glob #module wich create list of strings of a directory
import shutil #module which moves files
import os  # import os library
from astropy.io import ascii # import astropy library
from numpy.linalg import inv # import numpy library
import matplotlib.pyplot as plt  # import matplotlib library

from numpy.linalg import inv # import numpy library

import scipy as scipy # import scipy library
from scipy import optimize # import scipy library
from matplotlib.ticker import AutoMinorLocator  # import matplotlib library
from matplotlib import gridspec # import matplotlib library
import matplotlib.ticker as ticker  # import matplotlib library

#-------------------------------------PREAMBLE--------------------------------------------------
np.warnings.filterwarnings('ignore')

column_9 =  0 # Potential applied (V)
column_11 = 1 # WE(1).Current (A)
column_13 = 2 # Index



files = glob.glob('*.txt') #load every file with .txt extension 
print(files) #print in screen "files loaded"
#Raw text remarks
input_delimiter= "\t" #de raw data delimiter is tabular
input_skip_space=0 #we want 2 skip space from raw data
#Path in which we are working
path=os.getcwd() #get path 


scan_rate = [0.005,0.010,0.020,0.040,0.060] #list of scan rates

#-------------------------------------PREAMBLE Graphs--------------------------------------------------

SIZE_Font = 15 # Size fonts for plots
size=15 # Size fonts for plots
params = {'legend.fontsize': 'large',
          #'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)

label = ['5 mV/seg',"10 mV/seg",'20 mV/seg',"40 mV/seg","60 mV/seg"] #label set for graph
color = ['black','red','blue',"green","orange"] #colors set for graphs


#---------------------------------------------------------------------------------------
def new_directories(folder1,folder2):

    """This  function create new folders named as folder1 and folder2 in where will be stored figures and data separately and has been created 
    by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact"""


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

    """This  function remove every file in new folders named as folder1 and folder2  and has been created 
    by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact"""

    path= os.getcwd() + "/" #Obtain path 
    folder_1= folder1 #folder 1
    folder_2= folder2 #folder 2


    files_figures = glob.glob(path+folder1+ "/*")  #list all the files contained in folder 1
    for f in files_figures:  #for every file contained in folder 1
     os.remove(f)   #remove files

    files_data = glob.glob(path+folder2+ "/*") #list all the files contained in folder 2
    for f in files_data:   #for every file contained in folder 2
     os.remove(f)   #remove files

    print("-------------------------------------END REMOVING FILES IN FOLDERS FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def file_change_name(imput): #Use this function only to rename


    """This  function change the name of  files in order to obtain ordered list in our directory and has been created 
    by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact"""

    for name_old in imput: #for every file in our list 

        a=name_old[5:-6] # characters located between 3 and -7 position
        b = float(a) # convert characters selected to float
        c = '%04i' % b # add 0 up to 4 positions before numbers converted to float 
        print("The file name with 0 placed is: " + c)
        name_new= name_old.replace(a, c)
        print("Renamed file is: " + name_new)
        print(name_old, ' ------- ',   name_new)
        os.rename(name_old,name_new) #renamme  file
    print("-------------------------------------END CHANGE NAME OF OXYGEN FILES FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def ECSA():

    """This  function plot all the scan rates performed in electrochemical surface area analysis  and has been created 
    by Gaspar Carrasco. gasparcarrascohuertas@gmail.com for contact"""

    files = glob.glob(path +'**/**/data_mean_CV*.txt',recursive=True) #load every file with .txt extension 
    print(files) #print in screen "files loaded"
    counter=0 #counter set to 0
    fig, ax = plt.subplots() #subplot name as ax

    for file in files:   # For each file in files

        f = np.genfromtxt(file, delimiter="\t") #charge the file with tabular delimiter and skip header of 1 line
        WE_potential = f[:, 0] #Load from the "file" the column asociated to WE potential (V)
        WE_current = f[:, 1]  #Load from the "file" the column asociated to WE current (A)
        WE_current_corrected=WE_current #change WE current to microamperes

        ax.plot(WE_potential,WE_current_corrected,color=color[counter],label=label[counter]) #Plot the "WE potential (V)" vs  "WE current (A)"
        counter=counter+1 #we add 1 to counter for colors and labels

    #--------------------PLOTS OPTIONS--------------------------
 
    ax.tick_params(axis="y", right=True, direction='in')  #ticks Y-axe
    ax.tick_params(axis="x",top=True , direction='in')  #ticks X-axe
    ax.legend(loc='lower right', prop={'size':12}) #graph plot legend
    ax.set_xlabel('Potential vs. Ag/AgCl , 3.5 M (V)',  fontsize= SIZE_Font)  #graph plot X-Label
    ax.set_ylabel('Current intensity (\u00B5A) ',  fontsize= SIZE_Font)   #graph plot Y-Label

    ax.set_xlim(-0.1,1.2) # X-axe limits
    ax.set_ylim(-100,100)  # Y-axe limits
    ax.figure.savefig("figure_ECSA_GC.png") #Save plot as .png
    ax.figure.savefig("figure_ECSA_GC.eps") #Save plot as .eps

    shutil.move("figure_ECSA_GC.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_ECSA_GC.eps",  "figures") #Move the graph as "name.png" to folder figures
    #plt.show() # Show graph 
    #plt.clf() #clear the plot
    
    print("-------------------------------------END PLOT ECSA---------------------------------------------------")

def operation():

    files_ox = glob.glob(path +'**/**/data_info_table_oxidation*.txt',recursive=True) #load every file with .txt extension 
    print(files_ox) #print in screen "files loaded"

    files_red = glob.glob(path +'**/**/data_info_table_reduction*.txt',recursive=True) #load every file with .txt extension 
    print(files_red) #print in screen "files loaded"
    list1=[]
    list2=[]
    list_Epa=[]
    list_Epc=[]


    for file in files_ox:

        f1 = np.genfromtxt(file, delimiter=" ", skip_header=1)
        #print(f1)
        intersect_oxidation = f1[2]
        #print(intersect_oxidation)
        max_oxidation_peak = f1[1]
        intensity_oxidation_peak = max_oxidation_peak - intersect_oxidation


        Epa = f1[0]
        print("Potential anodic peak  (V):  " +str(Epa))
        print("Intensity anodic peak current (microA):  " +str(intensity_oxidation_peak))
        list1.append(intensity_oxidation_peak)
        list_Epa.append(Epa)

    for file in files_red:

        f2 = np.genfromtxt(file, delimiter=" ", skip_header=1)
        intersect_reduction = f2[2]
        max_reduction_peak = f2[1]
        intensity_reduction_peak = max_reduction_peak - intersect_reduction
        Epc = f2[0]
        print("Potential cathodic peak  (V):  " +str(Epc))
        print("Intensity cathodic peak current (microA):  " +str(intensity_reduction_peak))
        list2.append(intensity_reduction_peak)
        list_Epc.append(Epc)
  
    array_1=np.array(list1)
    np.savetxt("data_ipa.txt", array_1.transpose())
    array_Epa=np.array(list_Epa)
    np.savetxt("data_Epa.txt", array_Epa.transpose())

    array_2=np.array(list2)
    np.savetxt("data_ipc.txt", array_2.transpose())
    array_Epc=np.array(list_Epc)
    np.savetxt("data_Epc.txt", array_Epc.transpose())



    


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

    fig, ax1 = plt.subplots() #subplot named as ax1
    fig, ax2 = plt.subplots()  #subplot named as ax2
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    list_1 = []   #empty list  
    list_2 = []   #empty list  

    #--------------------------------------CATHODIC PROCESS-------------------------------------

    f_1 = np.genfromtxt("data_ipc.txt") #load file named  "data_ipc.txt" as numpy array
    ipc = np.array(f_1)
    ip_1 = ipc  # Cathodic peak current in microamps (microA)
    ip_corrected_1=ip_1*0.000001 #Cathodic  peak current  in amps (A) 

    list_1=(ipc,ip_corrected_1) #list with anodic peak current in amps and in microamps
    array_1=np.array(list_1) #array of list 
    tabla_1= Table(array_1.transpose()) #table of array
    new_column_1=Column(scan_rate, name="Scan rate (V/seg)")
    tabla_1.add_column(new_column_1) #add new column 
    tabla_1.write("data_info_table_oxidation_global.txt", format='ascii')

    f_2 = np.genfromtxt("data_info_table_oxidation_global.txt", delimiter=" ", skip_header=1) #load file named  "data_info_table_oxidation_global.txt" as numpy array with tab deliminter and skip first row
    scan_rate_list_1 = f_2[:, 2]  #scan rate in V/s
    print("scan_rate_list_1"+  str(scan_rate_list_1))

 
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
    ip_2 = ipa  # Cathodic peak current in microamps (microA)
    ip_corrected_2=ip_2*0.000001 #Cathodic  peak current  in amps (A) 

    list_2=(ipa,ip_corrected_2) #list with anodic peak current in amps and in microamps
    array_2=np.array(list_2) #array of list 
    tabla_2= Table(array_2.transpose()) #table of array
    new_column_2=Column(scan_rate, name="Scan rate (V/seg)")
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
    Area_final_cathodic=(a_1)/((262513.2907)*(1)*(C)*((D**0.5))) #calculate the area
    print("Area (cm2) Anodic " + str(Area_final_cathodic))



    text_file_1 = open("data_area_cathodic_convolution.txt", "w")
    n_1 = text_file_1.write(str(Area_final_cathodic))
    text_file_1.close()


    #--------------------------------------ANODIC PROCESS-------------------------------------

    f_3 = np.genfromtxt("data_ipa.txt")
    ipa = np.array(f_3)
    ip_2 = ipa  # Cathodic peak current in microamps (microA)
    ip_corrected_2=ip_2*0.000001 #Cathodic  peak current  in amps (A) 

    list_2=(ipa,ip_corrected_2) #list with anodic peak current in amps and in microamps
    array_2=np.array(list_2) #array of list 
    tabla_2= Table(array_2.transpose()) #table of array
    new_column_2=Column(scan_rate, name="Scan rate (V/seg)")
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

    #-------------------- AREA CALCULATING--------------------------
    Area_final_anodic=abs((a_2)/((262513.2907)*(1)*(C)*((D**0.5)))) #calculate the area
    print("Area (cm2) Cathodic " + str(Area_final_anodic))

    text_file_2 = open("data_area_anodic_convolution.txt", "w")
    n_2 = text_file_2.write(str(Area_final_anodic))
    text_file_2.close()


    #--------------------PLOTS--------------------------


    ax1.plot((scan_rate_list_1**0.5),ip_corrected_1,"o", color='black') #"plot root square of scan rate" vs "ipc"
    ax2.plot((scan_rate_list_2**0.5),ip_corrected_2,"o", color='red') #"plot root square of scan rate" vs "ipc"


    #--------------------PLOTS--------------------------

    ax1.plot((scan_rate_list_1**0.5),ip_corrected_1,"o", color='black') #"plot root square of scan rate" vs "ipc"
    ax2.plot((scan_rate_list_2**0.5),ip_corrected_2,"o", color='red') #"plot root square of scan rate" vs "ipc"

    #--------------------PLOTS OPTIONS--------------------------
    #--------------------ax1 PLOT--------------------------

    ax1.tick_params(axis="y", right=True, direction='in', labelcolor="red")     #ticks Y-axe
    ax1.tick_params(axis="x",top=True , direction='in')     #ticks X-axe
    #ax.legend(loc='lower right', prop={'size':8}) #graph plot legend
    ax1.set_xlabel(r"Scan rate $v^\frac{1}{2}$ (V/s) ",  fontsize= SIZE_Font)   #graph plot X-Label
    ax1.set_ylabel('Anodic peak  (Ipa , A)',  fontsize= SIZE_Font,  color="red")   #graph plot Y-Label
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    ax1.set_xlim(0,0.5)   #X-axe limits
    ax1.set_ylim(-0.0001,0.0001)  # Y-axe limits
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  #ticks Y-axe

    #--------------------ax2 PLOT--------------------------

    ax2.tick_params(axis="y", right=True, direction='in',  labelcolor="black") #ticks Y-axe
    ax2.tick_params(axis="x",top=True , direction='in') #ticks X-axe
    #ax.legend(loc='lower right', prop={'size':8}) #graph plot legend
    ax2.set_xlabel(r"Scan rate $v^\frac{1}{2}$ (V/s) ",  fontsize= SIZE_Font)
    ax2.set_ylabel('Cathodic peak  (Ipc , A)',  fontsize= SIZE_Font,  color="black")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    ax2.set_xlim(0,0.5) # X axe limits
    ax2.set_ylim(-0.0001,0.0001)  # Y axe limits
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  #ticks Y-axe

    fig.tight_layout()  # otherwise the right y-label is slightly clipped


    ax1.figure.savefig("figure_randles_sevick_plot_ECSA_GCE_3mm.png",bbox_inches='tight')  #Save plot as .png
    ax1.figure.savefig("figure_randles_sevick_plot_ECSA_GCE_3mm.eps",bbox_inches='tight')  #Save plot as .eps
    shutil.move("figure_randles_sevick_plot_ECSA_GCE_3mm.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_randles_sevick_plot_ECSA_GCE_3mm.eps",  "figures") #Move the graph as "name.png" to folder figures

    plt.clf() #clear the plot
    

    print("-------------------------------------END RANDLES-SEVICK FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
def alpha_aparent():

    n = 1   #number of electrons transferred in the redox event (usually 1)
    #A  == electrode area in cm2
    #F = 96485.33289  #Faraday Constant in C mol−1 == 96485.33289 mol−1
    D = 0.0000076  #diffusion coefficient in cm2/s == A 25°C los valores Do  (coeficiente de difusión) para el K_3 [Fe(CN)_6] en KCl 0,10 M son 7.6∗〖10〗^(−6) cm2 / s, respectivamente
    C = 0.000001  #concentration in mol/cm3 ----0.001 mol/L
    R = 8.3144598   #Gas constant in J K−1 mol−1  ==8.3144598 J⋅mol−1⋅K−1
    T = 298  #temperature in K = 25ºC ==298 K
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    f_1 = np.genfromtxt("data_Epa.txt", delimiter=" ", skip_header=0)
    Epa = np.array(f_1)
    print("Epa is "+  str(Epa))

    f_2 = np.genfromtxt("data_ipa.txt", delimiter=" ", skip_header=0)
    Ipa = np.array(f_2)
    absolute_ipa =np.abs(Ipa)
    logarithm_ipa= np.log(absolute_ipa)
    print("ln/Ipa/ is "+  str(logarithm_ipa)) 

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



    print(lsqfity(logarithm_ipa,Epa))
    print("slope ,  y-intercept,  correlation coefficient, standard deviation of the slope ,standard deviation of the y-intercept")  
    #ln(abs(Ip)) = ln abs(n*F*A*kap0*c0)   − (αap*n*F/R*T)*Ep

    ax1.plot(Epa,logarithm_ipa,"o", color='black') #"plot root square of scan rate" vs "ipc"
    ax2.plot(Epa,Ipa,"o", color='black') #"plot root square of scan rate" vs "ipc"

    ax1.tick_params(axis="y", right=True, direction='in',  labelcolor="black")
    ax1.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='lower right', prop={'size':8}) #graph legend
    ax1.set_xlabel("Ep",  fontsize= SIZE_Font)
    ax1.set_ylabel('ln|Ipa| (A)',  fontsize= SIZE_Font,  color="black")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    #ax1.set_xlim(0.34,0.38)
    #ax1.set_ylim(0,2.5)  # Y axe limits
    #plt.grid()# paint a grid over the graph
    # Change the y ticklabel format to scientific format
    #ax1.set_ylim(0,0.0005)  # Y axe limits
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax2.tick_params(axis="y", right=True, direction='in',  labelcolor="black")
    ax2.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='lower right', prop={'size':8}) #graph legend
    ax2.set_xlabel("Ep (V)",  fontsize= SIZE_Font)
    ax2.set_ylabel('Ipa (A)',  fontsize= SIZE_Font,  color="black")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    #ax2.set_xlim(0.34,0.38)
    #ax2.set_ylim(0,2.5)  # Y axe limits
    #plt.grid()# paint a grid over the graph
    # Change the y ticklabel format to scientific format
    #ax1.set_ylim(0,0.0005)  # Y axe limits
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    ax1.figure.savefig("figure_Apparent_transfer_coefficient.png",bbox_inches='tight')
    ax1.figure.savefig("figure_Apparent_transfer_coefficient.eps",bbox_inches='tight')
    shutil.move("figure_Apparent_transfer_coefficient.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_Apparent_transfer_coefficient.eps",  "figures") #Move the graph as "name.png" to folder figures

    ax2.figure.savefig("figure_Ipa_vs_Ep.png",bbox_inches='tight')
    ax2.figure.savefig("figure_Ipa_vs_Ep.eps",bbox_inches='tight')
    shutil.move("figure_Ipa_vs_Ep.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_Ipa_vs_Ep.eps",  "figures") #Move the graph as "name.png" to folder figures
    print("-------------------------------------END Alpha aparent anodic---------------------------------------------------")

def alpha_aparent_cathodic():

    n = 1   #number of electrons transferred in the redox event (usually 1)
    #A  == electrode area in cm2
    #F = 96485.33289  #Faraday Constant in C mol−1 == 96485.33289 mol−1
    D = 0.0000076  #diffusion coefficient in cm2/s == A 25°C los valores Do  (coeficiente de difusión) para el K_3 [Fe(CN)_6] en KCl 0,10 M son 7.6∗〖10〗^(−6) cm2 / s, respectivamente
    C = 0.000001  #concentration in mol/cm3 ----0.001 mol/L
    R = 8.3144598   #Gas constant in J K−1 mol−1  ==8.3144598 J⋅mol−1⋅K−1
    T = 298  #temperature in K = 25ºC ==298 K
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    f_1 = np.genfromtxt("data_Epc.txt", delimiter=" ", skip_header=0)
    Epc = np.array(f_1)
    print("Epa is "+  str(Epc))

    f_2 = np.genfromtxt("data_ipc.txt", delimiter=" ", skip_header=0)
    Ipc = np.array(f_2)
    absolute_ipc =np.abs(Ipc)
    logarithm_ipc= np.log(absolute_ipc)
    print("ln/Ipc/ is "+  str(logarithm_ipc)) 

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



    print(lsqfity(logarithm_ipc,Epc))
    print("slope ,  y-intercept,  correlation coefficient, standard deviation of the slope ,standard deviation of the y-intercept")  
    #ln(abs(Ip)) = ln abs(n*F*A*kap0*c0)   − (αap*n*F/R*T)*Ep

    ax1.plot(Epc,logarithm_ipc,"o", color='black') #"plot root square of scan rate" vs "ipc"
    ax2.plot(Epc,Ipc,"o", color='black') #"plot root square of scan rate" vs "ipc"

    ax1.tick_params(axis="y", right=True, direction='in',  labelcolor="black")
    ax1.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='lower right', prop={'size':8}) #graph legend
    ax1.set_xlabel("Ep",  fontsize= SIZE_Font)
    ax1.set_ylabel('ln|Ipa| (A)',  fontsize= SIZE_Font,  color="black")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    #ax1.set_xlim(0.34,0.38)
    #ax1.set_ylim(0,2.5)  # Y axe limits
    #plt.grid()# paint a grid over the graph
    # Change the y ticklabel format to scientific format
    #ax1.set_ylim(0,0.0005)  # Y axe limits
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax2.tick_params(axis="y", right=True, direction='in',  labelcolor="black")
    ax2.tick_params(axis="x",top=True , direction='in')
    #ax.legend(loc='lower right', prop={'size':8}) #graph legend
    ax2.set_xlabel("Epc (V)",  fontsize= SIZE_Font)
    ax2.set_ylabel('Ipc (A)',  fontsize= SIZE_Font,  color="black")
    #plt.title("Cyclic voltammetry \n Electrode Area")# graph title
    #ax2.set_xlim(0.34,0.38)
    #ax2.set_ylim(0,2.5)  # Y axe limits
    #plt.grid()# paint a grid over the graph
    # Change the y ticklabel format to scientific format
    #ax1.set_ylim(0,0.0005)  # Y axe limits
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    ax1.figure.savefig("figure_Apparent_transfer_coefficient_cathodic.png",bbox_inches='tight')
    ax1.figure.savefig("figure_Apparent_transfer_coefficient_cathodic.eps",bbox_inches='tight')
    shutil.move("figure_Apparent_transfer_coefficient_cathodic.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_Apparent_transfer_coefficient_cathodic.eps",  "figures") #Move the graph as "name.png" to folder figures

    ax2.figure.savefig("figure_Ipc_vs_Epc.png",bbox_inches='tight')
    ax2.figure.savefig("figure_Ipc_vs_Epc.eps",bbox_inches='tight')
    shutil.move("figure_Ipc_vs_Epc.png",  "figures") #Move the graph as "name.png" to folder figures
    shutil.move("figure_Ipc_vs_Epc.eps",  "figures") #Move the graph as "name.png" to folder figures
    print("-------------------------------------END Alpha aparent cathodic---------------------------------------------------")

def Deltap():

    """ This function calculate delta p , known as difference between oxidation an reduction peak potentials for all scan analysis (5 mV/seg)
    and     has been created by Gaspar, gasparcarrascohuertas@gmail.com for contact
    """

    f_1 = np.genfromtxt("data_ipa.txt")
    ppc = np.array(f_1)  #make array from data
 
    f_2 = np.genfromtxt("data_ipc.txt")
    ppa = np.array(f_2)  #make array from data

    potential_oxidation= f_1[0] #data associated to potential of oxidation peak 
    potential_reduction = f_2[0] #data associated to potential of reduction peak 

    substraction =  ppc -  ppa #substraction operation
    print("DeltaP value for all scan rate analysis is: "+str(substraction))
    print("-------------------------------------END DeltaP value---------------------------------------------------")

def move_data():

    """ This function move every data file .txt to folder data and
    has been created by Gaspar, gasparcarrascohuertas@gmail.com for contact
    """

    files = glob.glob(path+ "/data*.txt") #list every file named as data in .txt format
    print("Files which are going to be moved to directories are: "+str(files))
    for f in files: 
     shutil.move(f, "data") #Move file to folder named as data 
    print("-------------------------------------END MOVING DATA .TXT  FUNCTION---------------------------------------------------")
#---------------------------------------------------------------------------------------
new_directories("figures", "data")
removing_files_in_folder("figures","data")
#file_change_name(files)
ECSA()
operation()
Randles_Sevick_equation()
alpha_aparent()
alpha_aparent_cathodic()
Deltap()


move_data()

exit()








