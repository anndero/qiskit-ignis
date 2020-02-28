# -*- coding: utf-8 -*-
 
# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Class of Circuits for Bell Nonlocality testing."""

import qiskit
import itertools
from typing import List, Optional, Sequence, Union
from nptyping import Array

NumType = Union[int, float, complex]
GateType = Union[qiskit.circuit.Gate, qiskit.quantum_info.Operator, 
                 List[List[NumType]], np.ndarray]


class BLocCircuits:
    """ Class of Circuits for Bell Non-Locality testing.
    
    A typical Bell experiment requires a circuit preparing a n-qubit quantum state and 
    a set of 1-qubit unitary gates which precede measurements of each qubit. Assume that 
    on each qubit :math:`i` there is :math: `s_i` different projective measurements possible. 
    This is described by setting scenario: :math: `sett = [s_0,...,s_{n-1}]` and should be 
    reflected in the structure of passed pre_meas_gates argument. The list of qiskit circuits
    is build based on established gates and scenario in order due to increasing input indices.    
    
    Example of a qiskit Bell experiment:
        Construct a quantum state that will be tested, for example 2-qubits Bell 
        :math: `|\phi +\rangle` state:
    
        .. jupyter-execute::

            state_circ = qiskit.QuantumCircuit(2, name='phi+')
            state_circ.h(0)
            state_circ.cx(range(1), range(1, 2))
            
        Prepare a set of random pre-measurement gates according to setting scenario: 
        :math: `sett = [2,3]`:
        
        .. jupyter-execute::

            bloc = BLocCircuits.random([2,3])
            bloc.pre_meas_gates

        Construct a list of qiskit circuits ordered respectively to the following inputs order:
        :math: `(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)`:
        
        .. jupyter-execute::

            bloc.construct_meas_circs()
            bloc.meas_circs

        Finally perform a set of qiskit experiments and collect the results:
        
        .. jupyter-execute::

            shots = 100
            backend = qiskit.BasicAer.get_backend('qasm_simulator')
            results = [qiskit.execute(state_circ + circ, backend, shots).result() 
                       for circ in bloc.meas_circs]   
    """
    
    
    def __init__(self, pre_meas_gates: List[List[GateType]]):
        """ Initialize BLocCircuits suitable for Bell nonlocality test.

        Structure of pre_meas_gates should reflect the measurement settings scenario with 
        sublists of the available 1-qubit gates for each respective subsystem. For example:
        >>> self.pre_meas_gates = [[XGate,YGate,ZGate],[RXGate(pi/3),RYGate(pi/6)]]
        indicates 2 subsystems with 3 and 2 options respectively.               
        
        Args:
            pre_meas_gates (List[List[Gate or Operator or unitary matrix]]): 
                lists of grouped unitary gates preceding the measurements.
        """  
        self._gates = None
        self._circs = None
        self.pre_meas_gates = pre_meas_gates

        
    @property
    def pre_meas_gates(self) -> List[List[qiskit.extensions.UnitaryGate]]:
        """ Return the lists of the established unitary gates preceding the measurements. 
        
        The setter passes elements from the lists as UnitaryGate arguments.
        The setter clears previously constructed measurement circuits (self._circs = None).
        
        Returns:
            List[List[UnitaryGate]]: unitary gates."""
        return self._gates
        
        
    @pre_meas_gates.setter
    def pre_meas_gates(self, gates: List[List[GateType]]):
        self._gates = [[qiskit.extensions.UnitaryGate(i) for i in j] for j in gates]
        self._circs = None 
        
        
    @property
    def meas_circs(self) -> List[qiskit.QuantumCircuit]:
        """ Return the list of the circuits for Bell nonlocality testing in order due to increasing 
        input indices.

        Returns:
            List[QuantumCircuit]: measurement circuits."""
        return self._circs
        
        
    @property
    def gates_as_matrices(self) -> List[List[Array[complex, 2, 2]]]:
        """ Return unitary matrices of pre_meas_gates.
        
        Returns:
            List[List[Array[complex, 2, 2]]]: unitary matrices."""
        return [[i.to_matrix() for i in j] for j in self.pre_meas_gates] 
    

    @property
    def sett(self) -> List[int]:
        """ Return the number of possible measurement settings per subsystem."""
        return [len(i) for i in self.pre_meas_gates] 
        
        
    @property
    def n(self) -> int:
        """ Return the number of subsystems (i.e. number of qubits)."""
        return len(self.pre_meas_gates) 

    
    @classmethod
    def random(cls, sett: Sequence[int]):
        """ Construct BLocCircuits object with random unitary gates preceding the measurements.

        Args:
            sett (list[int]): possible measurement settings per subsystem. 
                len(sett) is the number of subsystems.
                
        Returns:
            BLocCircuits: instance with random pre measurement gates.
            
        Raises:
            QiskitError: if invalid sett argument.
        """
        if any(not isinstance(i, int) or i <= 0 for i in sett):
            raise qiskit.QiskitError('sett should be list of int > 0.')        
        pre_meas_gates = [[qiskit.extensions.UnitaryGate(
            qiskit.quantum_info.random.utils.random_unitary(2)) 
                           for s in range(sett[i])] for i in range(len(sett))]
        return BLocCircuits(pre_meas_gates)

    
    def construct_meas_circs(self) -> None:
        """ Construct the circuits for Bell nonlocality testing based on the established 
        pre measurement gates. The order of the circuits is due to increasing input indices.
        """
        meas_circs = []
        for inputs in itertools.product(*[range(s) for s in self.sett]):
            circ = qiskit.QuantumCircuit(self.n, self.n, name='{}'.format(inputs))
            circ.barrier()
            for i in range(self.n):
                circ.append(self.pre_meas_gates[i][inputs[i]], [i])
            circ.measure(range(self.n), range(self.n))
            meas_circs.append(circ)
        self._circs = meas_circs
