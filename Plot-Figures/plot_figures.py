import os, sys, re, yaml
import numpy as np
from operator import add
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Input data
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)

    fig_name = wano_file["figure-name"] # Same-graph 
    sub_fig_name = wano_file["subplot-name"] # Subplot
    plt.savefig(fig_name,dpi=200)
    plt.savefig(sub_fig_name,dpi=200)

    with open('Input-File') as file:
        data_file = yaml.full_load(file)

    colours=['r','orange','g','b','k','m','c','pink', 'brown', 'olive']

    if wano_file["Same-graph"]:
        n_plots = len(wano_file["Plots"])
        x_axis = []
        y_axis = []
        legends = []
        for ii in range(n_plots):
            x_axis.append(wano_file["Plots"][ii]["x-axis"])
            y_axis.append(wano_file["Plots"][ii]["y-axis"])
            legends.append(wano_file["Plots"][ii]["legend"])

        fig_title = wano_file["figure-title"]
        x_label = wano_file["x-label"]
        y_label = wano_file["y-label"]

        plt.figure() # In this example, all the plots will be in one figure.
        plt.suptitle(fig_title, size=16)    
        for ii in range(n_plots):
            plt.plot(data_file[x_axis[ii]],data_file[y_axis[ii]],colours[ii],marker="o", label = legends[ii])
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
        plt.savefig(fig_name,dpi=200)
        # plt.show()

    if wano_file["Subplot"]:
        n_subplots = len(wano_file["Subplots"])
        fig_title = wano_file["subplot-title"]
        x_axis = []
        y_axis = []
        x_label = []
        y_label = []

        for ii in range(n_subplots):
                x_axis.append(wano_file["Subplots"][ii]["x-axis"])
                y_axis.append(wano_file["Subplots"][ii]["y-axis"])
                x_label.append(wano_file["Subplots"][ii]["x-label"])
                y_label.append(wano_file["Subplots"][ii]["y-label"])

        fig, axs = plt.subplots(n_subplots,sharex=False, sharey=False)
        for ii in range(n_subplots):        
            axs[ii].plot(data_file[x_axis[ii]], data_file[y_axis[ii]],colours[ii], marker="o")
            axs[ii].set(ylabel = y_label[ii])
            axs[ii].set(xlabel = x_label[ii])

        plt.suptitle(fig_title, size=16)
        plt.subplots_adjust(hspace=0.8)
        plt.savefig(sub_fig_name,dpi=200)
        #plt.show()
