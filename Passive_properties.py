import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from implementation_HH import simulate_HH_neuron_adaptative
from implementation_HH import plot_Vm_I


def find_E_l(plot=True):
    """Function to find the parameter E_l"""
    # No input current to find E_l
    current = input_factory.get_zero_current()
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 100 * b2.ms)
    if(plot==True):
        plot_Vm_I(state_monitor_adaptative, r'Stimulation protocol: find parameter $E_l$.')
    
    #E_l corresponds to the constant potential value that is reached
    E_l =  max(state_monitor_adaptative.vm[0])
    if(plot==True):
        print("Parameter E_l = ", E_l)
    return E_l


def find_g_l(plot=True):
    """ Function to find parameter g_l"""
    # C'EST 1/R !!!

    g_l=1/find_R(False)

    if(plot==True):
        print("Parameter g_l =" , g_l)
    return g_l

def find_R(plot=True):
    """ Function to find parameter membrane resistance R"""
    # Need to input a step current: active for only one timestep
    # NOT A DELTA INPUT CURRENT LIKE IN CORRECTION LIF !!
    I_ext = 1.0
    current = input_factory.get_step_current(10, 80, b2.ms, I_ext * b2.uA)
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 100 * b2.ms)
    if(plot==True):
        plot_Vm_I(state_monitor_adaptative, r'Stimulation protocol: find parameter $R$.')

    v_max = max(state_monitor_adaptative.vm[0])
    v_rest = find_E_l(False)

    R = (v_max - v_rest)/(I_ext * b2.uA)
    if(plot==True):
        print("Parameter resistance of the membrane R =" , R)
    return R


def find_tau_m(plot=True):
    """ Function to find parameter membrane time constant tau_m"""
    # Tau_m corresponds to the time it takes for the neuron to reach 63% of max v_m value

    # Inject a step current such that the neuron doesn't fire
    I_ext = 1.0
    current = input_factory.get_step_current(0, 100, b2.ms, I_ext * b2.uA)
    state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 100 * b2.ms)
   
    if(plot==True):
        # Plot the figure with tau line
        fig,ax = plt.subplots(2,1)
        ax[0].plot(state_monitor_adaptative.t / b2.ms, state_monitor_adaptative.vm[0] / b2.mV, lw=2)
        ax[0].set_xlabel("t [ms]")
        ax[0].set_ylabel(r"$V_m$ [mV]")
        ax[0].grid()
        ax[0].hlines(y=-63.77, xmin=0, xmax=9.0, linewidth=1.8, color='g')
        ax[0].vlines(x=9.0, ymin=-70, ymax=-63.77, linewidth=2, color='g')
        ax[1].plot(state_monitor_adaptative.t / b2.ms, state_monitor_adaptative.I_e[0] / b2.uamp, "red", lw=2)
        ax[1].set_xlabel("t [ms]")
        ax[1].set_ylabel("$I_{e}$ [$\mu$ A]")
        ax[1].grid()
        plt.suptitle( r'Stimulation protocol: find parameter $\tau_m$.')
        plt.show()


    Vm_start = -70*b2.mV
    Vm_const = max(state_monitor_adaptative.vm[0])
    Vm_63 = Vm_start-(Vm_start-Vm_const)*0.63
    
    #idx_63 = np.where(state_monitor_adaptative.vm[0]==-63.77*b2.mV)[0]
    #print(idx_63)
    #tau_m = state_monitor_adaptative.t[idx_63]
    tau_m = 9.0 * b2.ms
    if(plot==True):
        print("63% of the maximal voltage = ",Vm_63)
        print("Parameter time constant of the membrane tau_m =" , tau_m)
    return tau_m

def find_C(plot=True):
    """ Function to find parameter C """
    # C= tau_m/R
    tau_m =find_tau_m(False)
    R = find_R(False)
    C=tau_m/R

    if(plot==True):
        print("Parameter C =" , C)
    return C


if __name__ == "__main__":
    find_E_l()
    find_g_l()
    find_R()
    find_tau_m()
    find_C()