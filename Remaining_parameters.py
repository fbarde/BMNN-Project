import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from neurodynex3.adex_model import AdEx
from implementation_HH import simulate_HH_neuron_adaptative
from adaptation import figure_adaptative
from scipy.signal import find_peaks


def spike_timings_AdEx(state_monitor) : 
    """
    Function extracting spike timing from voltage trace of AdEx neuron
    """

    # Get the membrane voltage of AdEx neuron
    Volt= state_monitor.v[0] / b2.mV
    time = state_monitor.t / b2.ms

    #Get the spike timings
    time_spike = []
    indice_spike = find_peaks(Volt, height= -40)
    time_spike = time[indice_spike[0]]

    # Get the time between two consecutive spikes
    diff_spiking_time = []
    for i in range(len(time_spike)-1) :
        diff_spiking_time.append(time_spike[i+1]-time_spike[i])

    return time_spike,diff_spiking_time



def figure_adaptative_AdEx(state_monitor,several_plot = False) : 
    
    """
    Function to plot the adaptative behavior of AdEX neuron.
    """

    time_spike,diff_spiking_time = spike_timings_AdEx(state_monitor)
    if several_plot == False : # case where i just have one value of coefficient a,b,c respectively.
        plt.figure()
    plt.plot(time_spike[1:],diff_spiking_time, 'r--')
    plt.xlabel(r"$t$ [ms]")
    plt.ylabel(r"$\Delta t$ [ms]")
    plt.grid()
    if several_plot == False : # case where i just have one value of coefficient a,b,c respectively.
        plt.show()



def find_remaining_param():

    """
    Function used to find the last parameters that would allow to match 
    the spike trains of AdEx model and HH adaptative neuron.
    """
    # Set the remaining parameters here:
    # Need to find parameters that match the two curves
    V_RESET= -77.2
    B= 45.35 
    TAU_W= 295 

    ## HH adaptative neuron model : biological ground truth to match
    current = input_factory.get_step_current(0, 1500, b2.ms, 2.0*b2.uA)
    HH_state_monitor_adaptive = simulate_HH_neuron_adaptative(current, simulation_time=1500 * b2.ms)
    
    # Adex model : need to find the right parameters:
    AdEx_state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=1500 * b2.ms, tau_m=9.0* b2.ms, 
    R=9.49628545* b2.kohm, v_rest=-70.60737437 * b2.mV, v_rheobase=-55.75543484 * b2.mV, delta_T= 1.9633764705215981* b2.mV , a=10655.855814888468 * b2.nS, 
    tau_w= TAU_W* b2.ms, v_spike = 30 * b2.mV, b= B* b2.nA, v_reset= V_RESET* b2.mV)  
    
    
    # Plot of the voltage traces:
    plt.figure()
    plt.plot(HH_state_monitor_adaptive.t / b2.ms, HH_state_monitor_adaptive.vm[0] / b2.mV, '-b', label='HH adaptive neuron')
    plt.plot(AdEx_state_monitor.t / b2.ms, AdEx_state_monitor.v[0] / b2.mV, '-r', label='AdEx neuron model')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('t [ms]')
    plt.ylabel(r'$V_m$ [mV]')
    plt.suptitle(r"Parameters: $b=$" + f"{B} [nA], " + r"$\tau_w =$" + f"{TAU_W} [ms], " + r"$V_{reset}=$" + f"{V_RESET} [mV]")


    # Plot of the adaptative behavior figures:
    plt.figure()
    legend =[]
    figure_adaptative(HH_state_monitor_adaptive,several_plot = True)
    legend.append('HH adaptive neuron') 
    figure_adaptative_AdEx(AdEx_state_monitor,several_plot = True)
    legend.append('AdEx neuron model') 
    plt.legend(legend) 
    plt.grid()
    plt.title("Adaptative behavior of HH adaptative neuron and AdEx model.")

    plt.show()

    print(f'The remaining parameters values are: \n V_reset = {V_RESET} mV \n b = {B} nA \n tau_w = {TAU_W} ms')




if __name__ == "__main__":

    find_remaining_param()
    