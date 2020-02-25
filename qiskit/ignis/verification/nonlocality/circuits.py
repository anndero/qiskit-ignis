""" Class of Circuits for Bell Nonlocality testing """

import qiskit
import numpy as np
import itertools as it

class BLocCircuits():
    """ Class of Circuits for Bell Nonlocality testing."""
    
    def __init__(self, pre_meas_gates):
        """ Constructor.
        Args:
            pre_meas_gates (list[list[Gate or Operator or unitary matrix]]): 
                lists of grouped unitary gates preceding the measurements.
                
        Additional Information:
        -----------------------
        Structure of pre_meas_gates should reflect the settings scenario 
        with sublists of available 1-qubit gates for a respective subsystem.
        For example:
        pre_meas_gates = [[XGate,YGate,ZGate],[RXGate(pi/3),RYGate(pi/6)]]
        indicates 2 subsystems with 3 and 2 options respectively.               
        """  
        self._gates = None
        self._circs = None
        self.pre_meas_gates = pre_meas_gates
            
    @property
    def pre_meas_gates(self):
        """ Return the lists of the established unitary gates preceding the 
            measurements.  
        Returns:
            list[list[UnitaryGate]]: unitary gates."""
        return self._gates
        
    @pre_meas_gates.setter
    def pre_meas_gates(self, gates):
        self._gates = [[qiskit.extensions.UnitaryGate(i) 
                        for i in j] for j in gates]
        self._circs = None # clear previously constructed circs
        
    @property
    def meas_circs(self):
        """ Return the list of the circuits for Bell nonlocality testing.
            The order of the circuits is due to increasing input indices.
        Returns:
            list[QuantumCircuit]: measurement circuits."""
        return self._circs
        
    @property
    def matrices(self):
        """ Return unitary matrices of pre_meas_gates.
        Returns:
            list[list[np.array]]: unitary matrices."""
        return [[i.to_matrix() for i in j] for j in self.pre_meas_gates] 

    @property
    def sett(self) -> list:
        """ Return the number of possible measurement settings per subsystem."""
        return [len(i) for i in self.pre_meas_gates] 
        
    @property
    def n(self) -> int:
        """ Return the number of subsystems (i.e. number of qubits)."""
        return len(self.pre_meas_gates) 

    @classmethod
    def random(cls, sett):
        """ Construct BLocCircuits object with random unitary gates preceding 
            the measurements.
        Args:
            sett (list[int]): possible measurement settings per subsystem.
                len(sett) is the number of subsystems.
        Returns:
            BLocCircuits: instance with random pre measurement gates.
        Raises:
            QiskitError: if invalid sett argument.
        """
        if any( not isinstance(i, int) or i <= 0 for i in sett ):
            raise qiskit.QiskitError('sett should be list of int > 0.')
        
        pre_meas_gates = [[qiskit.extensions.UnitaryGate(
            qiskit.quantum_info.random.utils.random_unitary(2))
                           for s in range(sett[i])] for i in range(len(sett))]
        return BLocCircuits(pre_meas_gates)

    def construct_meas_circs(self):
        """ Construct the circuits for Bell nonlocality testing
            based on the established pre measurement gates.
            The order of the circuits is due to increasing input indices.
        Returns:
            list[QuantumCircuit]: list of circuits. """
        meas_circs = []
        for inputs in it.product(*[range(s) for s in self.sett]):
            circ = qiskit.QuantumCircuit(self.n,self.n, name='{}'.format(inputs))
            circ.barrier()
            for i in range(self.n):
                circ.append(self.pre_meas_gates[i][inputs[i]], [i])
            circ.measure(range(self.n), range(self.n))
            meas_circs.append(circ)
        self._circs = meas_circs




