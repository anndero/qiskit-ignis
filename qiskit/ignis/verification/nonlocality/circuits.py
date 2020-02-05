""" circuits for Bell Nonlocality testing """

import qiskit
import numpy as np
import itertools as it
from math import sqrt, pi, cos, sin, asin, log
from cmath import exp
import logging


def state_circ(state = None, n = None ):
    """ part of a circuit corresponding to the initial quantum state 
    Args: 
        state: i.e. QuantumCircuit, statevector or state name (e.g. 'ghz','w','dicke2')
               if None then random pure state is prepared
        n(int): number of qubits 
    Returns:
        state: QuantumCircuit object preparing the state 
        statevector: vector of state' amplitudes 
    """
    
    # 1. state passed as a circuit
    if isinstance(state,qiskit.QuantumCircuit): 
        job = qiskit.execute(state, qiskit.BasicAer.get_backend('statevector_simulator'))
        statevector = job.result().get_statevector()
                    
    # 2. random state
    elif state == None and n != None:
        statevector = qiskit.quantum_info.random.utils.random_state(2**n)
        state = qiskit.QuantumCircuit(n, n, name="random_state")
        state.initialize(statevector,range(n))
   
    # 3. state passed by name e.g. ghz, w, dicke3         
    elif isinstance(state,str) and n != None:
        
        if state.lower() == 'ghz': 
            statevector = np.array(([1/sqrt(2)]+[0]*(2**n-2))+[1/sqrt(2)])
            state = qiskit.QuantumCircuit(n, n, name='ghz')
            state.h(0)
            state.cx(range(n-1), range(1,n))
            
        elif state.lower()[0:5] == 'dicke':
            try: excitations = int(state[5:])
            except ValueError: return
            statevector = [1.0/sqrt(n) if bin(i).count('1') == excitations else 0.0 for i in range(2**n)]
            state = qiskit.QuantumCircuit(n, n, name='w')
            state.initialize(statevector,range(n))
            
        elif state.lower() == 'w': 
            statevector = [1.0 if bin(i).count('1') == 1 else 0.0 for i in range(2**n)]
            statevector = statevector/sqrt(sum(statevector))
            state = qiskit.QuantumCircuit(n, n, name='dicke{}'.format(excitations))
            state.initialize(statevector,range(n)) 
        else: 
            logging.error(' name of state not recognized')
            return
            
    # 4. state passed as statevector
    elif hasattr(state,'__iter__'):
        if n == None: n = int(log(len(state),2))    
        if np.array(state).shape == (2**n,) and np.isclose(sum(np.conj(i)*i for i in state),1.0):
            statevector = state.copy()
            state = qiskit.QuantumCircuit(n, n, name="state_circ")
            state.initialize(statevector,range(n))   
        else: 
            logging.error(' statevector is not properly normalized')
            return
    else: 
        logging.error( ' state not recognized')
        return 
    state.barrier()    
    return state, statevector

        
def meas_circs(sett = None, unitary_seq = None):
    """ list of circuits corresponding to measurements
    Args: 
        sett: measurement settings scenario i.e. number of choices of observables for each qubit
        unitary_seq: list of lists of unitary matrices/gates that rotates the z measurement basis
                     if None then random unitaries are drawn
    Returns:
        circs_list: list of measurement circuits for all joint observable choices 
        unitary_seq: list of lists of unitary matrices that rotates the measurement basis
    """
    
    # 1. random list of unitaries based on setting scenario
    if unitary_seq == None: 
        n = len(sett)
        unitary_seq = [[su2() for mi in range(sett[i])] for i in range(n)]
        
    # 2. read scenario from sequence    
    elif sett == None: 
        sett = [len(s) for s in unitary_seq]
        n = len(sett)
        
    else: # 3. check if sequence and scenario match each other
        assert sett == [len(s) for s in unitary_seq]
        n = len(sett)
    
    # at that point unitary_seq can be passed as a list of either unitary matrices or gates
    circs_list = []
    for mi in it.product(*[range(s) for s in sett]):
        circ = qiskit.QuantumCircuit(n,n)
        for i in range(n):
            gate = unitary_seq[i][mi[i]]
            if isinstance(gate,qiskit.circuit.Gate):
                circ.append(gate, [i])
            else:
                circ.u3(gate, [i]) 
        circ.measure(range(n),range(n))
        circs_list.append(circ)
    
    # function will return unitary_seq as a list of unitary matrices
    unitary_seq = [[np.linalg.inv(u.to_matrix()) if isinstance(u,qiskit.circuit.Gate) else np.linalg.inv(u) for u in uu] for uu in unitary_seq]
        
        
    return circs_list, unitary_seq


def su2(psi = None, chi = None, phi = None) -> np.ndarray:
    """ returns a special unitary matrix SU(2) based on Euler angle parametrization 
        from K.Zyczkowski and M.Kus, J. Phys. A: Math. Gen. 27 (1994) 
        if angles are not provided then returns random SU(2) matrix distributed with Haar measure """
    
    if psi == None:[psi] = np.random.uniform(0,2*pi,1)
    if chi == None: [chi] = np.random.uniform(0,2*pi,1)  
    if phi == None:
        [xi] = np.random.uniform(0,1.0,1)
        phi = asin(sqrt(xi))
    
    u = np.array([[cos(phi)*exp(1j*psi),sin(phi)*exp(1j*chi)],
        [-sin(phi)*exp(-1j*chi),cos(phi)*exp(-1j*psi)]])
    return u

