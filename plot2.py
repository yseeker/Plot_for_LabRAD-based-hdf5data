# !pip install -U -q ipywidgets
# pip install plotly
# !jupyter nbextension enable --py widgetsnbextension
# !jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.36
# pip install cufflinks 

#8volts*3*epsilon_0/(55 nm*elementary charge) in cm^-2
#sqrt(6.0516e-16 / 8 * sqrt(3)*2*1.4e+12)*180/pi

import labrad
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import scipy.io as sio
from sklearn import datasets


import time
import math
import os
import scipy

import slackweb
import requests

import cmocean

# import plotly
# import plotly.offline as py
# import plotly.graph_objs as go

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'
# from IPython.display import Image, display, HTML

# from plotly.offline import plot, iplot, init_notebook_mode
# import plotly.graph_objs as go
# import cufflinks as cf

# Initialization
cxn = labrad.connect()
DV = cxn.data_vault

sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--','xtick.direction': 'in','ytick.direction': 'in'})

#matplotlib.rc('font', serif='Arial') 
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 3
matplotlib.rcParams['ps.fonttype'] = 3
    


# import labrad
# import numpy as np
# import time

# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from numpy.random import randn
# import seaborn as sns
# from datetime import datetime
# from matplotlib.ticker import ScalarFormatter
# import math
# import os

# def generate_cmap(colors):
#     values = range(len(colors))

#     vmax = np.ceil(np.max(values))
#     color_list = []
#     for v, c in zip(values, colors):
#         color_list.append( ( v/ vmax, c) )
#     return LinearSegmentedColormap.from_list('custom_cmap', color_list)

# pyqt = generate_cmap(['gold','red', 'k','blue', 'aqua'])
# pyqt2 = generate_cmap(['dodgerblue','blue', 'k','red', 'darkorange'])
# pblue = generate_cmap(['darkmagenta', 'white', 'navy'])

# Initialization
cxn = labrad.connect()
DV = cxn.data_vault



#xscale = [5, -5]
#yscale = [2, 3, 4, 6, 7, 8]
#num = [1, 5, 7] >=1
# num = list[range(1, 6)]   (1, 2, 3, 4, 5)
# num = list[range(0, 10, 2)] (0, 2, 4, 6, 8)
#xaxis = 3
#yaxis =  [5, 6, 7]

def oneD_plot2(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save, savename, xname, yname1, yname2):

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    sns.set('talk', 'whitegrid', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
    cr=iter(pyqt2(np.linspace(0,1,len(num))))
    color=iter(plt.cm.jet(np.linspace(0,1,len(num))))
    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        data = DV.get()
        c = next(cr)
        cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
        df = pd.DataFrame(data, columns=cl)
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(10, 16), label=str(j), c = c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(10, 16),label=str(j), c = c)

    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])

    
    ax1.legend().set_visible(True)
    ax2.legend().set_visible(False)


def oneD_plot3(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, xsize, ysize):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        data = DV.get()
        c = cr[j%10]
        cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

        df = pd.DataFrame(data, columns=cl)
        df[yaxis[1]]=abs(df[yaxis[1]])
        df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)#9, 14
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    #plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #plt.rcParams['svg.fonttype'] = 'none'

    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
             plt.savefig(savename+'.eps',transparent=True)
                #plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
def oneD_plot3_n2D(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, offset, factor, xsize, ysize):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        data = DV.get()
        c = cr[j%10]
        cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

        df = pd.DataFrame(data, columns=cl)
        df['n2D']=(df['Vg']-offset)*factor
        df[yaxis[1]]=abs(df[yaxis[1]])
        df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images'+file_name)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images'+file_name+'_'+savename+'.eps',transparent=True)
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        

def oneD_plot3_offset(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, offset, factor, offaxis, xsize, ysize):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        data = DV.get()
        c = cr[j%10]
        cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

        df = pd.DataFrame(data, columns=cl)
        df[offaxis]=(df[offaxis]-offset)*factor
        df[yaxis[1]]=abs(df[yaxis[1]])
        df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_path+'_'+savename+'.eps',transparent=True)
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

    
def line_plot(slope, magrange, offset, ax):
    x = np.linspace(magrange[0], magrange[1], 10)
    y = slope*(x-offset)
    ax.plot(x, y, color = "red")

def Fan_plot(FF, magrange, offset, ax):
    for i in range(len(FF)):
        slope0 = FF[i]
        line_plot(slope0, magrange, offset, ax)



        
def twoD_plot(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    df_pivot.to_csv(file_path+"landau_fan"+'.dat',mode='a', sep='\t')
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)

    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    
    
    if save == True:
        try:
            plt.savefig(savename+'.png', transparent=True)
            #plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png', transparent=True)
        except Exception:
            pass
    return fig, ax1
                   
                   

        
def twoD_plot_diff(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    #display(df_pivot)
    
    df_pivot = df_pivot.diff(1)
    
    df_pivot.to_csv(file_path+"landau_fan"+'.dat',mode='a', sep='\t')
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$dR/dn$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)

    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    
    
    if save == True:
        try:
            plt.savefig(savename+'.png', transparent=True)
            #plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png', transparent=True)
        except Exception:
            pass
    return fig, ax1



def twoD_plot_n2D(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, offset, factor, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df['n2D']=(df['vg']-offset)*factor
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    #fig = plt.figure(figsize=(12,6))
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    
    
    if save == True:
        try:
            print 'a'
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=300,transparent=True)
            print 'b'
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
    return fig, ax1


def twoD_plot_offset(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, offset, factor, offaxis, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df[offaxis]=(df[offaxis]-offset)*factor
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(12,6))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
#     for axis in ['top','bottom','left','right']:
#         ax1.spines[axis].set_linewidth(0.5)
        
#     for axis in ['top','bottom','left','right']:
#         ax1.spines[axis].set_visible(False)
        
#     plt.tick_params(labelbottom=False,
#                 labelleft=False,
#                 labelright=False,
#                 labeltop=False)
#     ax1.set_xticklabels([]) 
#     ax1.set_yticklabels([])
    plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.png', bbox_inches='tight', dpi=1000,transparent=True)

    
    
#     if save == True:
#         try:
#             plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=300,transparent=True)
            
#             plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.png', bbox_inches='tight', dpi=500,transparent=True)


def twoD_plot_offset_eps_png(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, offset, factor, offaxis, xsize, ysize, eps_png):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df[offaxis]=(df[offaxis]-offset)*factor
    df[values] = df[values]
    
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(12,6))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    
    
    x = np.arange(0, 1.0, 1.0)
    y = np.arange(0, 1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = X+Y

    if eps_png == "png":
        #c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        #cbar = fig.colorbar(c, ax=ax1)

        #cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    
    elif eps_png == "eps":
        c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        #c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        cbar = fig.colorbar(c, ax=ax1)

        cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
        
    #cbar = plt.colorbar()
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    if eps_png == "eps":
        ax1.set_xlabel(xname)
        ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        
    if eps_png == "png":
        
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_visible(False)

        ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        ax1.set_xticklabels([]) 
        ax1.set_yticklabels([])
        
    if eps_png == "eps":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'
    elif eps_png == "png":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.png', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'

            
            
def twoD_plot_offset_diff(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname,yname,offset,factor,offaxis,xsize,ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df[offaxis]=(df[offaxis]-offset)*factor
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = df_pivot.diff(1, axis=1)
    
    #df_pivot = abs(df_pivot.diff(1))
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(12,6))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)

            
def twoD_plot_offset_diff(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, offset, factor, offaxis, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    df[offaxis]=(df[offaxis]-offset)*factor
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = df_pivot.diff(1)
    
    #df_pivot = abs(df_pivot.diff(1))
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(12,6))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)

        
    return fig, ax1


def twoD_plot_ONE(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, value, axis, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    df1 = df[df[axis].isin([value])]
    #df1[offaxis]=(df1[offaxis]-offset)*factor
    df_pivot = pd.pivot_table(data=df1, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    
    
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    #fig.tight_layout()
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        
        
def twoD_plot_ONE_eps_png(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, value, axis, xsize, ysize, eps_png):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    df1 = df[df[axis].isin([value])]
    #df1[offaxis]=(df1[offaxis]-offset)*factor
    df_pivot = pd.pivot_table(data=df1, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    
    
    x = np.arange(0, 1.0, 1.0)
    y = np.arange(0, 1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = X+Y

    if eps_png == "png":
        #c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        #cbar = fig.colorbar(c, ax=ax1)

        #cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    
    elif eps_png == "eps":
        c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        #c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        cbar = fig.colorbar(c, ax=ax1)

        cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
        
    #cbar = plt.colorbar()
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    if eps_png == "eps":
        ax1.set_xlabel(xname)
        ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        
    if eps_png == "png":
        
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_visible(False)

        ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        ax1.set_xticklabels([]) 
        ax1.set_yticklabels([])
    
    if eps_png == "eps":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'
    elif eps_png == "png":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.png', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'
        
        
        
        
        
def twoD_plot_ONE_offset_eps_png(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, value, axis, offset, factor, offaxis, xsize, ysize, eps_png):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    df1 = df[df[axis].isin([value])]
    df1[offaxis]=(df1[offaxis]-offset)*factor
    
    df_pivot = pd.pivot_table(data=df1, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    x = np.arange(0, 1.0, 1.0)
    y = np.arange(0, 1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = X+Y

    if eps_png == "png":
        #c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        #cbar = fig.colorbar(c, ax=ax1)

        #cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    
    elif eps_png == "eps":
        c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        #c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        cbar = fig.colorbar(c, ax=ax1)

        cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
        
    #cbar = plt.colorbar()
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    if eps_png == "eps":
        ax1.set_xlabel(xname)
        ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        
    if eps_png == "png":
        
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_visible(False)

        ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        ax1.set_xticklabels([]) 
        ax1.set_yticklabels([])
    
    if eps_png == "eps":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'
    elif eps_png == "png":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.png', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'
        
        
def twoD_plot_sym_offset(file_path, file_name, num1, num2, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, sym, offset, factor, offaxis, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data1 = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df1 = pd.DataFrame(data1, columns=cl)
    
    a = df[values]
    b = df1[values]
    
    if(sym == True):
        df1['sym'] = (a+b)/2
    else:
        df1['sym'] = (a-b)/2
        
    df1[offaxis]=(df1[offaxis]-offset)*factor
    df_pivot = pd.pivot_table(data=df1, values='sym', columns=columns, index=index, aggfunc=np.mean)
    
    #df_pivot = df_pivot.diff(1)
    
    #df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize, ysize))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)

    
def twoD_plot_offset_diff_sym(file_path, file_name, num1, num2, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, sym, offset, factor, offaxis, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data1 = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df1 = pd.DataFrame(data1, columns=cl)
    
    a = df[values]
    b = df1[values]
    
    if(sym == True):
        df1['sym'] = (a+b)/2
    else:
        df1['sym'] = (a-b)/2
        
    df1[offaxis]=(df1[offaxis]-offset)*factor
    df_pivot = pd.pivot_table(data=df1, values='sym', columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = df_pivot.diff(1)
    
    
    #df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize, ysize))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)

    
    
        
def twoD_plot_sym_offset_linecut(file_path, file_name, num1, num2, values0, columns, index, vmin, vmax, cm, save, savename, xaxis, yaxis, xscale, yscale, norm, xname, yname, axis, values, sym, offset, factor, offaxis, xsize, ysize, logy0, fig, ax1):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df2 = pd.DataFrame(data, columns=cl)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data1 = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df3 = pd.DataFrame(data1, columns=cl)
    
    a = df2[values0]
    b = df3[values0]
    
    if(sym == True):
        df3['sym'] = (a+b)/2
    else:
        df3['sym'] = (a-b)/2
        
    df3[offaxis]=(df3[offaxis]-offset)*factor
    df0 = df3
    #df_pivot = pd.pivot_table(data=df1, values='sym', columns=columns, index=index, aggfunc=np.mean)
    
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in range(len(values)):
        c = cr[int(j)%10]

        vv = df0[axis].iloc[df0[axis].sub(values[j]).abs().idxmin()]
        print vv
        df = df0[df0[axis].isin([vv])]
        
        #display(df.head())
        
        df.plot(x=xaxis, y=yaxis, logy=logy0, ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    plt.rcParams["font.family"] = "sans-serif"
    
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])

    
    #ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)

    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)

    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)

    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
    #sns.heatmap(df_pivot)
    
    
def twoD_plot_sym_asym_offset(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, sym, offset, factor, offaxis, xsize, ysize):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    a = df[values]
    display(a)
    
    df1 = df.sort_values(['field_index', 'voltage_index'], ascending=[False, True])
    display(df1)

    
    b = df1[values]
    display(b)
    
    if(sym == True):
        df1['sym'] = (a+b)/2
    else:
        df1['sym'] = (a-b)/2
    display(df1['sym'])
    df1[offaxis]=(df1[offaxis]-offset)*factor

    df_pivot = pd.pivot_table(data=df1, values='sym', columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize, ysize))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        
    

    
def line_cut(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, values, axis, xsize, ysize):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
        
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df0 = pd.DataFrame(data, columns=cl)
    
    #display(df0)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in range(len(values)):
        c = cr[int(j)%10]

        vv = df0[axis].iloc[df0[axis].sub(values[j]).abs().idxmin()]
        print vv
        df = df0[df0[axis].isin([vv])]
        
        #display(df.head())
        
        df[yaxis[1]]=abs(df[yaxis[1]])
        df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        

        
def line_cut_scale(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, values, axis, offset, factor, offaxis, xsize, ysize):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df0 = pd.DataFrame(data, columns=cl)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in range(len(values)):
        c = cr[int(j)%10]

        vv = df0[axis].iloc[df0[axis].sub(values[j]).abs().idxmin()]
        print vv
        df = df0[df0[axis].isin([vv])]
        df[offaxis]=(df[offaxis]-offset)*factor
        display(df.head())
        
        df[yaxis[1]]=abs(df[yaxis[1]])
        df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass

        
    
def oneD_matrix3(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save, savename):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
#     cr=iter(pyqt2(np.linspace(0,1,20)))
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    for j in range(20):
        #c = next(cr)
        df2 = df[df['field_index'].isin([float(j)])]
        df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c='black')
        df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c='r')
        df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax3, figsize=(10, 20),label=str(j), c='b')
                   
#         df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c=c)
#         df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c=c)
#         df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax3, figsize=(10, 20),label=str(j), c=c)
    
    
    ax1.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax1.set_ylabel('$I$ (A)')
    ax2.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax2.set_ylabel('$R$'+'$\mathregular{_{xx}}$'+' (Ohm)')
    ax3.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax3.set_ylabel('$R$'+'$\mathregular{_{xy}}$'+' (Ohm)')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)
    
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        
        
def oneD_matrix_beta3(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save, savename, val):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #cr=iter(pyqt2(np.linspace(0,1,1)))
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    for j in range(20):
        #c = next(cr)
        df2 = df[df['Vg'].isin([val])]
        df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c='black')
        df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c="r")
        df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c="b")
    
    
    ax1.set_xlabel('$V$'+'$\mathregular{_{bias}}$'+' (V)')
    ax1.set_ylabel('$I$ (A)')
    ax2.set_xlabel('$V$'+'$\mathregular{_{bias}}$'+' (V)')
    ax2.set_ylabel('$V$'+'$\mathregular{_{2omega}}$'+' (V)')
    ax3.set_xlabel('$V$'+'$\mathregular{_{bias}}$'+' (V)')
    ax3.set_ylabel('$V$'+'$\mathregular{_{2omega}}$'+' (V)')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)
    
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in')
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    
    
def oneD_matrix3_sym(file_path, file_name, num1, num2, xaxis, yaxis, xscale, yscale, logy0, save, savename):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
#     cr=iter(pyqt2(np.linspace(0,1,20)))
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data1 = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df0 = pd.DataFrame(data1, columns=cl)
    
    for j in range(20):
        #c = next(cr)
        df3 = df[df['field_index'].isin([float(j)])]
        df4 = df0[df0['field_index'].isin([float(j)])]
        
        a1 = df3[yaxis[1]]
        a2 = df3[yaxis[2]]
        
        a3 = df4[yaxis[1]]
        a4 = df4[yaxis[2]]
        
        df4['Hall']=abs((a1+a3)/2)
        df4['n2d'] = (a2+a4)/2
    
    
#         df[yaxis[1]]=abs(df[yaxis[1]])
#         df[yaxis[2]]=abs(df[yaxis[2]])
        df4.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(9, 14), c="black")
        df4.plot(x=xaxis, y='Hall', logy=logy0, ax=ax2, figsize=(9, 14), c="r")
        df4.plot(x=xaxis, y='n2d', logy=logy0, ax=ax3, figsize=(9, 14), c="b")
        
        
        
        
#         df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c='black')
#         df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c='r')
#         df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax3, figsize=(10, 20),label=str(j), c='b')
                   
#         df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c=c)
#         df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c=c)
#         df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax3, figsize=(10, 20),label=str(j), c=c)
    
    
    ax1.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax1.set_ylabel('$I$ (A)')
    ax2.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax2.set_ylabel('$R$'+'$\mathregular{_{xx}}$'+' (Ohm)')
    ax3.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax3.set_ylabel('$R$'+'$\mathregular{_{xy}}$'+' (Ohm)')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)
    
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    plt.axis('tight')
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        
def oneD_matrix3_sym_one(file_path, file_name, num1, num2, xaxis, yaxis, xscale, yscale, logy0, save, savename, j):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
#     cr=iter(pyqt2(np.linspace(0,1,20)))
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data1 = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df0 = pd.DataFrame(data1, columns=cl)
    

    df3 = df[df['field_index'].isin([float(j)])]
    df4 = df0[df0['field_index'].isin([float(j)])]

    a1 = df3[yaxis[1]]
    a2 = df3[yaxis[2]]

    a3 = df4[yaxis[1]]
    a4 = df4[yaxis[2]]

    df4['Hall']=(a1+a3)/2
    df4['n2d'] = (a2-a4)/2


#         df[yaxis[1]]=abs(df[yaxis[1]])
#         df[yaxis[2]]=abs(df[yaxis[2]])
    df4.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(9, 14), c="black")
    df4.plot(x=xaxis, y='Hall', logy=logy0, ax=ax2, figsize=(9, 14), c="r")
    df4.plot(x=xaxis, y='n2d', logy=logy0, ax=ax3, figsize=(9, 14), c="b")
        
        
        
        
#         df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c='black')
#         df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c='r')
#         df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax3, figsize=(10, 20),label=str(j), c='b')
                   
#         df2.plot(x=xaxis, y=yaxis[0], logy=logy0, ax=ax1, figsize=(10, 20), label=str(j), c=c)
#         df2.plot(x=xaxis, y=yaxis[1], logy=logy0, ax=ax2, figsize=(10, 20),label=str(j), c=c)
#         df2.plot(x=xaxis, y=yaxis[2], logy=logy0, ax=ax3, figsize=(10, 20),label=str(j), c=c)
    
    
    ax1.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax1.set_ylabel('$I$ (A)')
    ax2.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax2.set_ylabel('$R$'+'$\mathregular{_{xx}}$'+' (Ohm)')
    ax3.set_xlabel('$V$'+'$\mathregular{_{G}}$'+' (V)')
    ax3.set_ylabel('$R$'+'$\mathregular{_{xy}}$'+' (Ohm)')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)
    
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
    
    
def R_G(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save, savename, xname, yname1, yname2):

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    #cr=iter(pyqt2(np.linspace(0,1,len(num))))
    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        data = DV.get()
        #c = next(cr)
        cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
        df = pd.DataFrame(data, columns=cl)
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(9, 14), label=str(j),c='r')
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(9, 14),label=str(j), c='b')
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            
###### Plotly ##########
def oneD_plot3_plotly(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, xname, yname1, yname2, yname3):
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df0 = pd.DataFrame(data, columns=cl)

    # Create traces
    trace0 = go.Scatter(
        x = df0[xaxis],
        y = df0[yaxis[0]],
        mode = 'lines',
        name = 'lines'
    )
    trace1 = go.Scatter(
        x = df0[xaxis],
        y = df0[yaxis[1]],
        mode = 'lines',
        name = 'lines'
    )
    trace2 = go.Scatter(
        x = df0[xaxis],
        y = df0[yaxis[2]],
        mode = 'lines',
        name = 'lines'
    )

    fig = tools.make_subplots(rows=3, cols=1)

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace2, 3, 1)

    data = [trace0]
    axis_style=dict(zeroline=False, showline=True, mirror=True)
    #layout = go.Layout(title='Sample 6', width=650, height=500, xaxis = axis_style, yaxis=axis_style)
    #figure = go.Figure(data=data, layout=layout)
    fig['layout'].update(height=1000, width=650, title='Vg scan', xaxis = axis_style, yaxis=axis_style, xaxis2 = axis_style, yaxis2=axis_style, xaxis3 = axis_style, yaxis3=axis_style)

    xtitle = 'Vg (V)'

    fig['layout']['xaxis1'].update(title=xname, range=[xscale[0], xscale[1]])
    fig['layout']['yaxis1'].update(title=yname1, range=[xscale[0], xscale[1]])
    fig['layout']['xaxis2'].update(title=xname, range=[xscale[0], xscale[1]])
    fig['layout']['yaxis2'].update(title=yname2, range=[xscale[2], xscale[3]])
    fig['layout']['xaxis3'].update(title=xname, range=[xscale[0], xscale[1]])
    fig['layout']['yaxis3'].update(title=yname3, range=[xscale[4], xscale[5]])

    fig.layout.template = 'plotly_dark'
    iplot(fig)
    
    
    

def twoD_plot_plotly(file_path, file_name, num, values, columns, index, zmin, zmax, cm, xmin, xmax, ymin, ymax, norm, xname, yname):              

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df0 = pd.DataFrame(data, columns=cl)
    df_pivot = pd.pivot_table(data=df0, values=values, columns=columns, index=index, aggfunc=np.mean)

    data = [go.Heatmap(x=df_pivot.index, y=df_pivot.columns, z=df_pivot.T, colorscale=cm, zmin=zmin, zmax=zmax)]

    axis_style=dict(zeroline=False, showline=True, mirror=True)
    layout = go.Layout(title='Landau Fan', width=650, height=500, xaxis = axis_style, yaxis=axis_style)

    figure = go.Figure(data=data, layout=layout)
    figure.layout.template = 'plotly_dark'

    figure['layout']['xaxis'].update(title=xname, range=[xmin, xmax])
    figure['layout']['yaxis'].update(title=yname, range=[ymin, ymax])

    iplot(figure)
    
    
    
def Hall(file_path, file_name, num1, num2, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3,xsize,ysize):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
        
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a1 = df[yaxis[1]]
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a2 = df[yaxis[1]]
    
    df['Hall']=(a1-a2)/2/1.0
    df['n2d'] = 1/((a1-a2)/2)/1.6e-15/1.0
    
    
#         df[yaxis[1]]=abs(df[yaxis[1]])
#         df[yaxis[2]]=abs(df[yaxis[2]])
    df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), c="b")
    df.plot(x=xaxis, y='Hall', logy=logy0[1], ax=ax2, figsize=(xsize, ysize), c="b")
    df.plot(x=xaxis, y='n2d', logy=logy0[2], ax=ax3, figsize=(xsize, ysize), c="b")
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_path+'_'+savename+'.eps',transparent=True)
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
def Hall_offset(file_path, file_name, num1, num2, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3,offset, factor, offaxis,xsize,ysize):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
        
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a1 = df[yaxis[1]]
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    df[offaxis]=(df[offaxis]-offset)*factor
    
    a2 = df[yaxis[1]]
    
    df['Hall']=(a1-a2)/2/1.0
    df['n2d'] = 1/((a1+a2)/2)/1.6e-15/2.0
    
    
#         df[yaxis[1]]=abs(df[yaxis[1]])
#         df[yaxis[2]]=abs(df[yaxis[2]])
    df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), c="b")
    df.plot(x=xaxis, y='Hall', logy=logy0[1], ax=ax2, figsize=(xsize, ysize), c="b")
    df.plot(x=xaxis, y='n2d', logy=logy0[2], ax=ax3, figsize=(xsize, ysize), c="b")
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_path+'_'+savename+'.eps',transparent=True)
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
        
        
def sym(file_path, file_name, num1, num2, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
        
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a1 = df[yaxis[1]]
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a2 = df[yaxis[1]]
    
    df['Hall']=(a1-a2)/2
    df['n2d'] = (a1+a2)/2
    
    
#         df[yaxis[1]]=abs(df[yaxis[1]])
#         df[yaxis[2]]=abs(df[yaxis[2]])
    df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(9, 14), c="b")
    df.plot(x=xaxis, y='Hall', logy=logy0[1], ax=ax2, figsize=(9, 14), c="b")
    df.plot(x=xaxis, y='n2d', logy=logy0[2], ax=ax3, figsize=(9, 14), c="b")
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png')
        except Exception:
            pass
        
        
def Hallp(file_path, file_name, num1, num2, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3,xsize,ysize):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
        
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a1 = df[yaxis[1]]
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data = DV.get()

    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    df = pd.DataFrame(data, columns=cl)
    
    a2 = df[yaxis[1]]
    
    df['Hall']=(a1+a2)/2/1.0
    df['n2d'] = 1/((a1+a2)/2)/1.6e-15/1.0
    
    
#         df[yaxis[1]]=abs(df[yaxis[1]])
#         df[yaxis[2]]=abs(df[yaxis[2]])
    df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), c="b")
    df.plot(x=xaxis, y='Hall', logy=logy0[1], ax=ax2, figsize=(xsize, ysize), c="b")
    df.plot(x=xaxis, y='n2d', logy=logy0[2], ax=ax3, figsize=(xsize, ysize), c="b")
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png')
        except Exception:
            pass
    
    
def stack_2Dplot_scale(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, offset, factor, offaxis, xsize, ysize, logy):

    data = None
    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        if data is not None:
            data = np.vstack((data,DV.get()))
        else:
            data = DV.get()
            
            
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    

        
    
    
    df[offaxis]=(df[offaxis]-offset)*factor
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(12,6))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)

    #c = ax1.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    
    c.set_clip_path(ax1.patch)
    cbar = fig.colorbar(c, ax=ax1)
    #cbar = plt.colorbar()
    cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    if logy==True:
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    
    #plt.ylim(ymin, ymax)
    #plt.xlim(xmin, xmax)
    
    ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})

    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    
#     ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
#     for axis in ['top','bottom','left','right']:
#         ax1.spines[axis].set_linewidth(0.5)
    
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    
    
    if save == True:
        try:
            print 'a'
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=300,transparent=True)
            print 'b'
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass
        
    return fig, ax1



def stack_2Dplot_scale_eps_png(file_path, file_name, num, values, columns, index, vmin, vmax, cm, save, savename, xmin, xmax, ymin, ymax, norm, xname, yname, offset, factor, offaxis, xsize, ysize, logy, eps_png):

    data = None
    
    for j in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][j-1])
        if data is not None:
            data = np.vstack((data,DV.get()))
        else:
            data = DV.get()
            
            
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df = pd.DataFrame(data, columns=cl)
    

        
    
    
    df[offaxis]=(df[offaxis]-offset)*factor
    df[values]=df[values]
        
    df_pivot = pd.pivot_table(data=df, values=values, columns=columns, index=index, aggfunc=np.mean)
    
    df_pivot = abs(df_pivot)
    
    
    # to igor
    # path = fn+scan+"_matrix"+'.dat'
    # if os.path.exists(path) == False:
    #     df_pivot.to_csv(fn+scan+"matrix"+'.dat',mode='a', sep='\t')
    # df_pivot
    
    #ax = plt.subplot(111)
    fig = plt.figure(figsize=(xsize,ysize))
    #fig = plt.figure(figsize=(12,6))
    #fig = plt.figure(figsize=(7,10))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(111)
    if logy==True:
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    #ax = plt.subplot()
    #plt.pcolor(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    # cmap="jet, gnuplot, gnuplot2, magma, plasma, **inferno, seismic, " or 
    #ax.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T)

    #c = ax1.pcolorfast(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
    x = np.arange(0, 1.0, 1.0)
    y = np.arange(0, 1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = X+Y

    if eps_png == "png":
        #c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        #cbar = fig.colorbar(c, ax=ax1)

        #cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
    
    elif eps_png == "eps":
        c = ax1.pcolormesh(X, Y, Z, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        #c = ax1.pcolormesh(df_pivot.index, df_pivot.columns, df_pivot.T, cmap=cm, vmin=vmin, vmax=vmax, norm=norm)
        
        c.set_clip_path(ax1.patch)
        cbar = fig.colorbar(c, ax=ax1)

        cbar.set_label('$R$'+'$\mathregular{_{}}$'+' (Ohm)',size=18)
        
    #cbar = plt.colorbar()
    #plt.axis('tight')
    #plt.xlabel(xname)
    #plt.ylabel(yname)
    
    if eps_png == "eps":
        ax1.set_xlabel(xname)
        ax1.set_ylabel(yname)

    fig.patch.set_facecolor('white')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    
    #ax1.set_title(savename)
    
    sns.set('talk', 'ticks', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 1.0, 'grid.linestyle': '--'})
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        
    if eps_png == "png":
        
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_visible(False)

        ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        ax1.set_xticklabels([]) 
        ax1.set_yticklabels([])
    
    if eps_png == "eps":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'
    elif eps_png == "png":
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.png', bbox_inches='tight', dpi=1000,transparent=True)
        except Exception:
            print 'save failed'



def stack_line_cut_scale(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, values, axis, offset, factor, offaxis, xsize, ysize, fig, ax1, ax2, ax3):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(311)
#     ax2 = fig.add_subplot(312)
#     ax3 = fig.add_subplot(313)
    
    data = None
    
    for ii in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][ii-1])
        if data is not None:
            data = np.vstack((data,DV.get()))
        else:
            data = DV.get()
            
            
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    
    df0 = pd.DataFrame(data, columns=cl)
    
    
    
    
    #colors=cmocean.cm.balance(np.linspace(0,1,len(values)))
    colors=plt.cm.viridis(np.linspace(0,1,len(values)))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in range(len(values)):
        #c = cr[int(j)%10]
        c = cr[int(j)%10]
        c = colors[j]

        vv = df0[axis].iloc[df0[axis].sub(values[j]).abs().idxmin()]
        print vv
        df = df0[df0[axis].isin([vv])]
        df[offaxis]=(df[offaxis]-offset)*factor
        #display(df.head())
        
        df[yaxis[1]]=abs(df[yaxis[1]])
        #df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)

        
        #### Igor convert START
        
#         num_moji = num_moji.replace('[', '')
#         num_moji = num_moji.replace(']', '')
#         num_moji = num_moji.replace(' ', '')
#         num_moji = num_moji.replace(',', '_')
        if save == True:
        
            value_moji = str(values[j])
            value_moji = value_moji.replace('.', 'p')
            value_moji = value_moji.replace('-', 'm')

            fn = str(file_path)+'_'+str(num[0])+'_'+value_moji

            path_RB = fn+'.itx'
            col = []
            for ii in range (0, len(cl)):
                col.append(fn+'_'+cl[ii])
            igor_moji = ' '.join(col)

            if os.path.exists(path_RB) == False:
                print 'a'
                with open(fn+'.itx', "a") as f:
                    f.write("IGOR" + "\n" + "WAVES /O " + igor_moji + "\n" + "BEGIN" + "\n")
                    f.close

                df.to_csv(fn+'.itx',mode='a', header=False, index=False, sep='\t', columns=cl)
                with open(fn+'.itx', "a") as f:
                    f.write("END")
                    f.close
        #### Igor convert END
        
        
        
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    try:
        os.mkdir('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path)
        os.mkdir('Z://_personal//Yu//data_images//'+file_name)
    except Exception:
        pass
    
    if save == True:
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_path+'_'+savename+'.eps',transparent=True)
            plt.savefig('Z://_personal//Yu//data_images//'+file_name+'//'+file_name+'_'+savename+'.png',transparent=True)
        except Exception:
            pass



def stack_line_cut_scale_2axis(file_path, file_name, num, xaxis, yaxis, xscale, yscale, logy0, save,savename, xname, yname1, yname2, yname3, values, axis, offset, factor, offaxis, offset2, factor2, offaxis2, xsize, ysize, fig, ax1, ax2, ax3):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(311)
#     ax2 = fig.add_subplot(312)
#     ax3 = fig.add_subplot(313)
    
    data = None
    
    for ii in num:
        DV.cd('')
        DV.cd(file_path)
        DV.open(DV.dir()[1][ii-1])
        if data is not None:
            data = np.vstack((data,DV.get()))
        else:
            data = DV.get()
            
            
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    
    df0 = pd.DataFrame(data, columns=cl)
    
    
    
    #r=iter(plt.cm.brg(np.linspace(0,1,len(num))))
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in range(len(values)):
        c = cr[int(j)%10]

        vv = df0[axis].iloc[df0[axis].sub(values[j]).abs().idxmin()]
        print vv
        df = df0[df0[axis].isin([vv])]
        df[offaxis]=(df[offaxis]-offset)*factor
        df[offaxis2]=(df[offaxis2]-offset2)*factor2
        #display(df.head())
        
        df[yaxis[1]]=abs(df[yaxis[1]])
        df[yaxis[2]]=-df[yaxis[2]]
        #df[yaxis[2]]=abs(df[yaxis[2]])
        df.plot(x=xaxis, y=yaxis[0], logy=logy0[0], ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
        df.plot(x=xaxis, y=yaxis[1], logy=logy0[1], ax=ax2, figsize=(xsize, ysize),label=str(j), c=c)
        df.plot(x=xaxis, y=yaxis[2], logy=logy0[2], ax=ax3, figsize=(xsize, ysize),label=str(j), c=c)
        
        #### Igor convert START
        
#         num_moji = num_moji.replace('[', '')
#         num_moji = num_moji.replace(']', '')
#         num_moji = num_moji.replace(' ', '')
#         num_moji = num_moji.replace(',', '_')
        if save == True:
        
            value_moji = str(values[j])
            value_moji = value_moji.replace('.', 'p')
            value_moji = value_moji.replace('-', 'm')

            fn = str(file_path)+'_'+str(num[0])+'_'+value_moji

            path_RB = fn+'.itx'
            col = []
            for ii in range (0, len(cl)):
                col.append(fn+'_'+cl[ii])
            igor_moji = ' '.join(col)

            if os.path.exists(path_RB) == False:
                print 'a'
                with open(fn+'.itx', "a") as f:
                    f.write("IGOR" + "\n" + "WAVES /O " + igor_moji + "\n" + "BEGIN" + "\n")
                    f.close

                df.to_csv(fn+'.itx',mode='a', header=False, index=False, sep='\t', columns=cl)
                with open(fn+'.itx', "a") as f:
                    f.write("END")
                    f.close
        #### Igor convert END
        
        
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname1)
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname2)
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname3)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax2.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    ax3.legend().set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)
    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax2.tick_params(axis = 'both', direction = 'in', width = 0.5)
    ax3.tick_params(axis = 'both', direction = 'in', width = 0.5)
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
        ax3.spines[axis].set_linewidth(0.5)
    
    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
#     try:
#         os.mkdir('Z://_personal//Yu//data_images//'+file_name)
#     except Exception:
#         pass
    
    if save == True:
        try:
            plt.savefig('D://Dropbox//Young_Lab//personal//data_pdf_images//'+file_path+'//'+file_name+'_'+savename+'.eps', bbox_inches='tight', dpi=300,transparent=True)
        except Exception:
            pass

    

def onsagar_linecut_scale2axis(file_path, file_name, num1, num2, values0, columns, index, vmin, vmax, cm, save, savename, xaxis, yaxis, xscale, yscale, norm, xname, yname, axis, values, sym, offset, factor, offaxis, offset2, factor2, offaxis2, xsize, ysize, logy0, fig, ax1):

    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num1-1])
    data = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df2 = pd.DataFrame(data, columns=cl)
    
    DV.cd('')
    DV.cd(file_path)
    DV.open(DV.dir()[1][num2-1])
    data1 = DV.get()
    cl = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]
    df3 = pd.DataFrame(data1, columns=cl)
    
    a = df2[values0]
    b = df3[values0]
    
    if(sym == True):
        df3['sym'] = (a+b)/2
    else:
        df3['sym'] = (a-b)/2
        
    df3[offaxis]=(df3[offaxis]-offset)*factor
    df3[offaxis2]=(df3[offaxis2]-offset2)*factor2
    df0 = df3
    #df_pivot = pd.pivot_table(data=df1, values='sym', columns=columns, index=index, aggfunc=np.mean)
    
    cr = ["crimson", "darkblue", "deeppink", "darkorange", "dimgray","deepskyblue", "black", "darkcyan","brown", "blue"]    
    for j in range(len(values)):
        c = cr[int(j)%10]

        vv = df0[axis].iloc[df0[axis].sub(values[j]).abs().idxmin()]
        print vv
        df = df0[df0[axis].isin([vv])]
        
        #display(df.head())
        
        df.plot(x=xaxis, y=yaxis, logy=logy0, ax=ax1, figsize=(xsize, ysize), label=str(j),c=c)
    
    
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)

    plt.rcParams["font.family"] = "sans-serif"
    
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.tight_layout()
    #fig.patch.set_facecoloxr('white')
    ax1.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])

    
    #ax1.set_title(savename)
    
    ax1.legend().set_visible(False)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0,fontsize=18)

    
    ax1.tick_params(axis = 'both', direction = 'in', width = 0.5)

    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)

    
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    
