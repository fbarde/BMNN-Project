import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from Passive_properties import find_E_l, find_g_l
from implementation_HH import simulate_HH_neuron_adaptative
from scipy.stats import linregress



def plotI_V(current): 

    """
    Fuction to extract the parameter a from the I-V curve.
    """

    state_monitor = simulate_HH_neuron_adaptative(current, 10000 * b2.ms)
    plt.figure()
    #Plot the I-V curve 
    plt.plot(state_monitor.I_e[0][400:] / b2.uamp ,state_monitor.vm[0][400:] / b2.mV,lw=2)
    plt.xlabel("injected current I_ext [uA]",size=13)
    plt.ylabel("membrane potential V [mV]",size=13)
    plt.title("I-V curve",size = 14)
    plt.grid()
    plt.show()
    # Parameter a can be obtained from the slope of the I-V curve 
    x = state_monitor.I_e[0][400:]/ b2.uamp
    y = state_monitor.vm[0][400:] / b2.mV
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    gl = find_g_l(False)/b2.uS
    #Relation to get a from the slope
    a = 1/slope-gl*10**(-3)
    return a



if __name__ == "__main__":
    current = input_factory.get_ramp_current(0,10000 , b2.ms, 0 * b2.uA,1.2* b2.uA)
    a = plotI_V(current)
    print("The value of the coefficient a is :",a, "millisiemens")
    



