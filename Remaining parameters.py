from neurodynex3.adex_model import AdEx
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from implementation_HH import simulate_HH_neuron_adaptative


def find_remaining_param():

    #test_current = input_factory.get_step_current(0, 1500, b2.ms, 2.0*b2.uA)  #-0.9243 

    # Set the parameters here:
    V_RESET= -70
    B= 90
    TAU_W= 150

    ## HH model to copy 
    current = input_factory.get_step_current(0, 1500, b2.ms, 2.0*b2.uA)
    HH_state_monitor_adaptive = simulate_HH_neuron_adaptative(current, simulation_time=1500 * b2.ms)
    


    Adex_state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=1500 * b2.ms, tau_m=9.0* b2.ms, 
    R=9.49628545* b2.kohm, v_rest=-70.60737437 * b2.mV, v_rheobase=-55.75543484 * b2.mV, delta_T= 1.9633764705215981* b2.mV , a=10655.855814888468 * b2.nS, 
    tau_w= TAU_W* b2.ms, v_spike = 30 * b2.mV, b= B* b2.nA, v_reset= V_RESET* b2.mV)  
    
    
    
    plt.figure()
    plt.plot(HH_state_monitor_adaptive.t / b2.ms, HH_state_monitor_adaptive.vm[0] / b2.mV, '-b', label='HH adaptive neuron')
    plt.plot(Adex_state_monitor.t / b2.ms, Adex_state_monitor.v[0] / b2.mV, '-r', label='AdEx neuron')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('t [ms]')
    plt.ylabel(r'$V_m$ [mV]')
    plt.suptitle(r"Parameters: $b=$" + f"{B} [nA], " + r"$\tau_w =$" + f"{TAU_W} [ms], " + r"$V_{reset}=$" + f"{V_RESET} [mV]")

    plt.show()


if __name__ == "__main__":

    find_remaining_param()
    