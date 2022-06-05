from xml.etree import ElementPath
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from Passive_properties import find_E_l, find_g_l
from implementation_HH import plot_data, simulate_HH_neuron_adaptative,plot_Vm_I
from scipy.signal import find_peaks
from scipy.optimize import fsolve


def sketch_f_vs_V():  

    """
    Function to plot f(V) as a function of V
    """

    # Parameters needed in expression of f(V)

    E_l= -70.60737437#find_E_l(False)/ b2.mV
    g_l= 105.3043#find_g_l(False)/ b2.msiemens
    V_S= -51.26383235

    # Found values for theta and delta
    theta_rh= -55.75543484#find_theta_rh(False)/b2.mV
    delta_T= 1.9633764705215981#find_delta_T()/b2.ms

    # Range of membrane pot for which plot f
    V=np.linspace(-75,-50,50)
    
    f_v=np.zeros(len(V))

    # Formula of f(V)
    for i in range(len(V)):
        f_v[i] = -g_l*(V[i] - E_l) + g_l * delta_T*np.exp((V[i]-theta_rh)/delta_T) #+ I_ext

    # Plot f vs V and remarcable quantities
    plt.figure()
    plt.plot(V, f_v, lw=2)
    plt.hlines(y=0, xmin=-75, xmax=-45, colors='b', linestyles='dashed',lw=1.5)
    plt.vlines(x=E_l, ymin=-1500, ymax=2000, colors='k',linestyles='dashdot', lw=1.5, label=r'$E_l=-70.6073$ [mV]' )
    plt.vlines(x=theta_rh, ymin=-1500, ymax=2000, colors='g',linestyles='dashdot', lw=1.5, label=r'$\theta_{rh}=-55.7554$ [mV]' )
    plt.vlines(x=V_S, ymin=-1500, ymax=2000, colors='r',linestyles='dashdot', lw=1.5, label=r'$V_S=-51.2638$ [mV]' )
    plt.xlabel(r'Membrane potential $V_m$ [mV]')
    plt.ylabel(r'Membrane dynamics $f(V_m)$ [$\mu$A]')
    plt.suptitle('Membrane dynamics as a function of V, and remarquable parameters.')
    plt.legend()
    plt.grid()
    plt.show()


def find_theta_rh(plot=True):

    """
    Function to find parameter theta_rh.
    """
    
    # Step current for the minimum of f to be slightly above zero
    I_ext=1.5

    current = input_factory.get_step_current(0 ,50 , b2.ms, I_ext * b2.uA)
    state_monitor = simulate_HH_neuron_adaptative(current, 50 * b2.ms) # simulation time is short because not interested in the AP
    #plot_data(state_monitor, type='adaptative', title=None)
    
    # theta_rh is the voltage corresponding to the minimum of f(V)
    # f(V) is also equal to membrane_I_m (see equations of HH neuron)
    theta_rh = state_monitor.vm[0][np.argmin(state_monitor.membrane_Im[0]/b2.uamp)]

    if (plot==True):
        plt.figure()
        plt.plot(state_monitor.vm[0] / b2.mV, state_monitor.membrane_Im[0]/b2.uamp, lw=2, label=r'$I_{ext}$=1.5 [$\mu$A]')
            
        plt.xlabel(r"$V_m$ [mV]")
        plt.ylabel(r"$f(V_m)$ [$\mu$A]")
        plt.hlines(y=0, xmin=-75, xmax=-45, colors='b', linestyles='dashed',lw=1.5)
        plt.vlines(x=theta_rh/b2.mV, ymin=-0.5, ymax=7, linewidth=2, colors='g', linestyles='dashed', label=r'$\theta_{rh}=-55.75$ [mV]')
        plt.grid()
        plt.legend()
        plt.suptitle( r'Stimulation protocol: find parameter $\theta_{rh}$.')
        plt.show()

        
        print('The value of theta_rh is', theta_rh)

    return theta_rh
    


def find_Vs():

    """
    Function to find parameter V_S.
    """
    #Current amplitude for which the neuron voltage reach max without spiking  
    I = np.linspace(10.642, 10.6424, 10)
    for I_ext in I:
        
        #Inject short pulse current of 1 us
        current = input_factory.get_step_current(10 ,11 , b2.ms, I_ext * b2.uA)
        state_monitor = simulate_HH_neuron_adaptative(current, 50 * b2.ms)
        #plot_data(state_monitor, type='adaptative', title=None)
        Volt= state_monitor.vm[0] / b2.mV
        # look at the number of spike and take the last current and v_max before spikes start to appear
        indice_spike = find_peaks(Volt, height=0)
        print(indice_spike)
        print(I_ext)
        V_S = max(state_monitor.vm[0])
        print('V_s=',V_S)
        # In the end, take the last printed value that had no spike
        
def function_f_V(delta_T):

    """
    Equation f(V).
    """

    E_l= find_E_l(False)/ b2.mV
    g_l= find_g_l(False)/ b2.msiemens
    V_S= -51.26383235
    theta_rh =find_theta_rh(False)/ b2.mV

    return -g_l*(V_S - E_l) + g_l * delta_T*np.exp((V_S-theta_rh)/delta_T)



def find_delta_T():

    """
    Function to find delta_T.
    Solve f(V_s)=0.
    """

    delta_T = fsolve(function_f_V,[1.0, 5.0])
    print('Solution delta_T:' ,delta_T[0])
    return delta_T[0]



if __name__ == "__main__":
    find_theta_rh()
    find_Vs()
    find_delta_T()
    sketch_f_vs_V()


