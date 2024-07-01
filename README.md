# BMNN-Project
Computational modeling of single neurons: from Hodgkin-Huxley to AdEx

**Date:** Feb-June 2022\
**Course:** CS-433 - Biological Modelling of Neural Network (Prof. Wulfram Gerstner)\
**Collaborators:** Flore Barde, Jeannette Gommendy\

## General aim of the code

Neurons are cellular structures whose complex functioning is governed by many voltage-dependent
ions channels. Multiple theoretical models were developed over the years in order to model single
neurons, to better understand their intrinsic mechanisms. This code is part of a project aiming at investigating some models for the single neuron. As the starting point, the Hodgkin Huxley function will be implemented in order to get a spiking model. Two functions will be implemented for regular spiking (INa, IK currents) and adaptive spiking (INa, IK, IM currents). From these complex Hodgkin Huxley models, parameters will then be tuned out in order to develop a computationally less expensive model: the Adaptive Exponential Integrate and Fire (AdEx) neuron model.
The provided code gathers the different methods used, the functions used for their optimization as well as scripts to reproduce the obtained results.

## Structure of the code

Each exercise of the project has its own Python file, named corresponding to the title of the exercise.
Running the main of each Python file allows to print the values of the parameters found, and plot the figures asked.

**1.1 Getting started :** implementation_HH.py

**1.2 Rebound spike? :** rebound_spike.py (and implementation_2.py and rebound2.py for comparision with Exercise 5)

**1.3 Adaptation :** adaptation.py

**2.1 Passive properties :** Passive_properties.py

**2.2 Exponential Integrate and Fire :** Exponential_Integrate_and_Fire.py

**2.3 Subthreshold adaptation :** Subthreshold_adaptation.py

**2.4 Remaining parameters :** Remaining_parameters.py

**2.5 Testing on random input :** Random_input.py


