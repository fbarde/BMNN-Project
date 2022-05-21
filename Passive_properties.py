import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from implementation_HH import simulate_HH_neuron_adaptative
from implementation_HH import plot_Vm_I


def find_E_l():
    """Function to find the parameter E_l"""
    # No input current to find E_l
    current = input_factory.get_zero_current()
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 100 * b2.ms)
    plot_Vm_I(state_monitor_adaptative, r'Stimulation protocol: find parameter $E_l$.')
    
    #E_l corresponds to the constant potential value that is reached
    E_l =  max(state_monitor_adaptative.vm[0])
    print("Parameter E_l = ", E_l)
    return E_l

def find_g_l():
    """ Function to find parameter g_l"""

    # Inject a very small input current such that the neuron doesn't fire
    I_ext = 0.1
    current = input_factory.get_step_current(10, 100, b2.ms, I_ext * b2.uA)
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 100 * b2.ms)
    plot_Vm_I(state_monitor_adaptative, r'Stimulation protocol: find parameter $g_l$.')

    E_l = find_E_l()
    Vm_const = max(state_monitor_adaptative.vm[0])

    g_l = I_ext*b2.uA/(Vm_const - E_l)

    print("Parameter g_l =" , g_l)
    return g_l

def find_R():
    """ Function to find parameter membrane resistance R"""
    # Need to input a delta current: active for only one timestep
    I_ext = 1.0
    current = input_factory.get_step_current(10, 10, b2.ms, I_ext * b2.uA)
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 50 * b2.ms)
    plot_Vm_I(state_monitor_adaptative, r'Stimulation protocol: find parameter $R$.')

    v_max = max(state_monitor_adaptative.vm[0])
    v_rest = -69.7* b2.mV

    R = (v_max - v_rest)/(I_ext * b2.uA)
    print("Parameter resistance of the membrane R =" , R)
    return R


def find_tau_m():
    """ Function to find parameter membrane time constant tau_m"""
    # Tau_m corresponds to the time it takes for the neuron to reach 63% of max v_m value

    # Inject a step current such that the neuron doesn't fire
    I_ext = 1.0
    current = input_factory.get_step_current(0, 100, b2.ms, I_ext * b2.uA)
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 100 * b2.ms)
    plot_Vm_I(state_monitor_adaptative, r'Stimulation protocol: find parameter $\tau_m$.')

    Vm_start = -70*b2.mV
    Vm_const = max(state_monitor_adaptative.vm[0])
    Vm_63 = Vm_start-(Vm_start-Vm_const)*0.63
    print(Vm_63)
    #idx_63 = np.where(state_monitor_adaptative.vm[0]==-63.77*b2.mV)[0]
    #print(idx_63)
    #tau_m = state_monitor_adaptative.t[idx_63]
    tau_m = 9.0 * b2.ms
    print("Parameter time constant of the membrane tau_m =" , tau_m)
    return tau_m

def find_C():
    """ Function to find parameter C """
    # C= tau_m/R
    tau_m =find_tau_m()
    R = find_R()
    C=tau_m/R

    print("Parameter C =" , C)
    return C


if __name__ == "__main__":
    find_E_l()
    find_g_l()
    find_R()
    find_tau_m()
    find_C()