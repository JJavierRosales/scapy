#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:34:03 2022

@author: jjrr
"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from fractions import Fraction

#%%
def time_formatted(seconds):
    
    hours = int(seconds/3600)
    minutes = int((seconds - hours*3600)/60)
    seconds = int(seconds - hours*3600 - minutes*60)
    
    return str(hours) + "h " + str(minutes) +"min " + str(seconds) + "s"
#%%
#Function to remove outliers from a list
def remove_outliers(data, threshold):

    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = Q3-Q1

    data_cleaned = data[(data>=(Q1-IQR*threshold)) & 
                         (data<=(Q3+IQR*threshold))]
    
    return data_cleaned
#%%
#Function to calculate distance between two points
def calculate_distance(P,Q):
    P = np.asarray(P).astype(np.float)
    Q = np.asarray(Q).astype(np.float)
    return np.linalg.norm(P-Q)

#%%
def deg2rad(d):
    return d/360*2*np.pi

def rad2deg(r):
    return r/(2*np.pi)*360

#%%
def import_2Darray(file_path):
    
    with open(file_path) as textFile:
        output_array = [line.split(" ") for line in textFile]
    
    return np.array(output_array).astype(np.float)

#%%
def orbit_elems_info():
    
    #Array with the name of the orbital parameters
    params_dict = {"a":{"limits":[0,3.5],
                        "label": r"Semi-major axis ($a$)",
                        "units": None},
                   "e":{"limits":[0,1],
                        "label": r"Eccentricity ($e$)",
                        "units": None},
                   "om":{"limits":[0,2*np.pi],
                         "label": r"Longitude of ascending node ($\Omega$)",
                         "units": "radians"},
                   "w":{"limits":[0,2*np.pi],
                        "label": r"Argument of Periapsis ($\omega$)",
                        "units": "radians"},
                   "i":{"limits":[0,np.pi/8],
                        "label": r"Inclination ($i$)",
                        "units": None},
                   "q":{"limits":[0,1.3],
                        "label": r"Periapsis distance ($q$)",
                        "units": None}}
    
    return params_dict

#%%
def radian_axis_labels(x_lower, x_upper, intervals=4):
    x_tick = np.linspace(x_lower,x_upper,intervals+1)
    x_label = []
    
    for tick in (x_tick/np.pi):
        
        if tick%1==0:
            if tick==0:
                x_label.append(r"$" + str(int(tick)) + "$")
            elif tick==1:
                x_label.append(r"$\pi$")
            else:
                x_label.append(r"$" + str(int(tick)) + "\pi$")
        else:
            numerator = Fraction(tick).numerator \
                if Fraction(tick).numerator!=1 else ""
            denominator = Fraction(tick).denominator
            x_label.append(r"$\frac{" + str(numerator) + "\pi}{" + 
                           str(denominator) + "}$")

    return x_tick, x_label

#%%
def get_metrics(y_true, y_pred, show_metrics=True):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    err1 = fp/len(y_true)
    err2 = fn/len(y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    if show_metrics==True:
        
        print("\tTP: %s \tFP: %s" % (tp, fp))
        print("\tFN: %s \tTN: %s" % (fn, tn))
        
        print("\n\tError type I: %.2f%%" % round(err1*100,4))
        print("\tError type II: %.2f%%" % round(err2*100,4))
        
        # Data correctly classified
        print("\n\tAccuracy = %.2f%%" % round(accuracy*100,4)) 
        # Predicted Positives truly Positive
        print("\tPrecision = %.2f%%" % round(precision*100,4))
        # Actual Positives correctly classified
        print("\tRecall = %.2f%%" % round(recall*100,4))
        print("\tF1 = %.2f%%" % round(f1*100,4))
    
    return err1, err2, accuracy, precision, recall, f1

#%%
if __name__ == "__main__":
    
    test = orbit_elems_info()
    
    x_lower, x_upper = test["om"]["limits"]
    
    x_tick, x_label = radian_axis_labels(x_lower, x_upper)
    
    print(x_label)
    
    

    
    
    
    
    
    
    