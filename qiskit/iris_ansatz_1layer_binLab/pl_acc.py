#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

# Function to load and parse data
#...!...!....................
def load_data(file_path):
    data = []
    # content: expName,acc,iIter,loss,myRank,elaT,num_weight
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#sum"):
                parts = line.split(',')
                # Extract acc and elaT values (indices may vary based on your data structure)
                acc = float(parts[2])  # Assuming acc is the third element
                nIter=int(parts[3])
                loss = float(parts[4])  
                elaT = float(parts[6])  
                nWei = float(parts[7])  
                data.append((acc, nIter,loss,elaT,nWei))
    expName=parts[1]
    return data,expName

#...!...!....................
# Task function: Plot histogram
def plot_acc_vs_elaT(data):
    acc_values = [item[0] for item in data]
    elaT_values = [item[1] for item in data]

    plt.hist2d(acc_values, elaT_values, bins=[30, 30], cmap='Blues')
    plt.colorbar()
    plt.xlabel('Accuracy')
    plt.ylabel('Elapsed Time')
    plt.title('Histogram of Accuracy vs Elapsed Time')
    plt.show()


#...!...!....................
# New Task function: Plot 1D histogram of acc
def plot_acc(data):
    acc_values = [item[0] for item in data]
    nIterL = [item[1] for item in data]
    lossL = [item[2] for item in data]
    elaTL = [item[3]/60. for item in data]
    
    nTask=len(acc_values)
    # Compute mean and standard deviation
    mean_iter = np.mean(nIterL)
    std_iter = np.std(nIterL)
    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values)
    mean_loss = np.mean(lossL)
    std_loss = np.std(lossL)
    mean_elaT = np.mean(elaTL)
    std_elaT = np.std(elaTL)
    
    
    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(6,4))

    # Plotting the histogram
    ax.hist(acc_values, bins=10, color='green', alpha=0.7)
    ax.set_xlabel('Accuracy,   avr_Iter=%.d'%mean_iter)
    ax.set_ylabel('Frequency')
    ax.set_xlim(0.55,1.02)
    
    # Marking the mean and standard deviation
    ax.errorbar(mean_acc, ax.get_ylim()[1]/2, xerr=std_acc, fmt='o', color='red', label=f'Mean: {mean_acc:.2f}, Std: {std_acc:.2f}')

    txt='avr accuracy %.3f +/- %.3f '%(mean_acc, std_acc)
    txt+='\navr elaT(min) %.1f +/- %.1f '%(mean_elaT, std_elaT)
    txt+='\navr iter %.0f +/- %.0f '%(mean_iter, std_iter)
    txt+='\navr loss %.3f +/- %.3f '%(mean_loss, std_loss)
    txt+='\n nWeights=%d'%data[0][4]
    print('PL:',txt)
    ax.text(0.05,0.60,txt, transform = ax.transAxes,color='b')
    ax.legend()
    ax.grid()
    ax.set_title('Accuracy, ntask=%d, exp=%s, job= %s'%(nTask,expName, args.jid))
    plt.tight_layout()

    figName=os.path.join(args.path,args.jid+".png")
    print('Graphics saving to ',figName)
    plt.savefig(figName)
    plt.show()

    
#=================================
#  M A I N
#=================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting with Pyplot")
    parser.add_argument("--path", default='out', help="Path to the data directory")
    parser.add_argument("-j","--jid", default='18337621', help="Slurm job")
    args = parser.parse_args()

    dataFile='%s.out'%args.jid
    full_path = os.path.join(args.path, dataFile)
    print('input:',full_path)
    data,expName = load_data(full_path)

    #plot_histogram(data)
    plot_acc(data)
    
    
    #main()
