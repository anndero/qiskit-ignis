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

""" Bell nonlocality testing class."""

import qiskit
import cplex
import itertools
import functools
import collections
import numpy as np
from math import ceil

try:
    from matplotlib import pyplot as plt
    PLT = True
except ImportError:
    PLT = False

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell' or "google.colab._shell":
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm
except NameError:
    from tqdm import tqdm as tqdm

from qiskit.ignis.verification.nonlocality.bell_scenario import BellScenario
    
from typing import Optional, Sequence, Union, List, Dict
NumType = Union[int, float, complex]
StateType = Union[qiskit.quantum_info.states.DensityMatrix, 
                  qiskit.quantum_info.states.Statevector,
                  np.ndarray, List[List[NumType]]]
GateType = Union[qiskit.circuit.Gate, qiskit.quantum_info.Operator, 
                 List[List[NumType]], np.ndarray]


class BLocFitter(BellScenario):
    #-----------------------------------------------------------------------------------------------
    def __init__(self, 
                 sett: Sequence[int], 
                 results: Optional[List[qiskit.result.Result]] = None, 
                 state: Optional[StateType] = None, 
                 pre_meas_gates: Optional[List[List[GateType]]] = None):
        """ Initialize BLocFitter with attributes. 
        
        Assumes that results come from qiskit circuits built of state preparation part and 
        measurements preceded by pre_meas_gates.
    
        Providing state and pre_meas_gates is necessary to calculate theorethical tests. 
        Structure of pre_meas_gates should reflect the settings scenario with sublists of available 
        1-qubit gates or matrices for a respective subsystem. For example:
        >>> pre_meas_gates = [[XGate,YGate,ZGate],[RXGate(pi/3),RYGate(pi/6)]]
        indicates 2 subsystems with 3 and 2 options respectively.  
        
        Args:
            sett (Sequence[int]): possible measurement settings per subsystem. len(sett) is the 
                number of subsystems.    
            results (List[Result]): qiskit circuit results in order due to the increasing input 
                indices.
            state (Statevector or DensityMatrix or matrix or vector): a quantum state.
            pre_meas_gates (List[List[Gate or Operator or unitary matrix]]): lists of grouped 
                unitary gates preceding the measurements.
        """    
        
        super().__init__(sett=sett, d=2)

        self._meas_qcorr = []
        self._calc_qcorr = {}  

        self._results = []  
        if results is not None:
            self.add_results(results)  
        
        self._rho = None
        if state is not None:
            self.rho = state

        self._gates = None
        if pre_meas_gates is not None:
            self.pre_meas_gates = pre_meas_gates

        self._LP = None             
    
    
    # qiskit results --> measured quantum correlations
    #-----------------------------------------------------------------------------------------------
    @property
    def results(self) -> List[List[qiskit.result.Result]]:
        """ Return a list of all added qiskit experiment results.
        
        Returns: 
            List[List[Result]]: qiskit experiment results."""
        return self._results

    
    def add_results(self, results: List[List[qiskit.result.Result]], clear: bool = False):  
        """ Add list of qiskit results to self.results. Calculate measured quantum correlations 
        and update self.meas_qcorr.
        
        Args:
            results (List[Result]): qiskit circuits results in order due to the increasing input 
                indices.
            clear (bool): if True then clear all previously added results, otherwise all results 
                are taken into account [Default: False].   
                
        Raises: 
            TypeError: if elements of results list are not qiskit Results.
            ValueError: if results are incompatible with setting scenario or if all results 
                doesn't have the same number of shots.
        """
        if not all(isinstance(r, qiskit.result.Result) for r in results):
            raise TypeError("Results should be a list of qiskit Result objects.")    
        if len(results) != np.prod(self.s):
            raise qiskit.QiskitError(
                "Results should be a list of length {}".format(np.prod(self.s))) 
        if len(set([r.results[0].shots for r in results])) != 1:        
            raise qiskit.QiskitError(
                "Number of shots in all results in the list should be the same.")
        if clear:
            self._results = []
        self._results.append(results)
        
        self.calc_meas_qcorr()
        
    
    def calc_meas_qcorr(self) -> None:
        """ Calculate measured quantum correlations and set self.meas_qcorr."""
        
        total_counts = np.zeros(self.rows_a)
        shots = 0
        for results in self._results:
            shots += results[0].results[0].shots
            counts = []
            for res in results:
                counts.extend([v for k, v in sorted(res.get_counts().items())])
            total_counts += np.array(counts)
        self._meas_qcorr = total_counts/shots
             
            
    # density matrix of quantum state
    #-----------------------------------------------------------------------------------------------
    @property
    def rho(self) -> np.ndarray:
        """ Return the density matrix of a quantum state.
        
        The setter validates if an arg is a valid quantum state."""
        return self._rho 
    
    
    @rho.setter
    def rho(self, state: StateType):
        if not isinstance(state, qiskit.quantum_info.states.DensityMatrix):
            state = qiskit.quantum_info.states.DensityMatrix(state)
        if (state.is_valid() and len(state.dims()) == self.n and len(set(state.dims())) == 1):
            self._rho = state.data
            self._d = state.dims()[0]
        else: 
            raise qiskit.QiskitError("Invalid state.")

            
    # unitary matrices of pre measurement gates
    #-----------------------------------------------------------------------------------------------
    @property
    def pre_meas_gates(self) -> List[List[np.ndarray]]:
        """ Return the lists of the established unitary gates preceding the measurements.  
        
        Returns:
            List[List[np.ndarray]]: unitary matrices."""
        return self._gates
    
        
    @pre_meas_gates.setter
    def pre_meas_gates(self, gates: List[List[GateType]]):
        if (len(gates) != self.n or any(len(gates[i]) != self.s[i] for i in range(self.n))): 
            raise qiskit.QiskitError(
                "pre_meas_gates are incompatible with the settings scenario.")
        self._gates = [[self.__check_single_gate(i) for i in j] for j in gates]
        self._calc_qcorr.clear()  
        
        
    def __check_single_gate(self, gate: GateType) -> np.ndarray:
        
        if hasattr(gate, 'to_matrix'):
            gate = gate.to_matrix()
        elif hasattr(gate, 'to_operator'):
            gate = gate.to_operator().data
        gate = np.array(gate, dtype=complex)

        if not qiskit.quantum_info.operators.predicates.is_unitary_matrix(gate):
            raise qiskit.QiskitError("Matrix is not unitary.") 
        if gate.shape[0]!= self.d:
            raise qiskit.QiskitError("Matrix is incompatible with self.d.")
        return gate
             
        
    # quantum correlations
    #-----------------------------------------------------------------------------------------------
    def get_quantum_corr(self, source: str, row: int) -> np.float:
        """ Return quantum correlation for inputs and outputs of a given row. 
        
        Args: 
            source (str): source of quantum correlations:
                'meas':  probabilities from qiskit experiment results.
                'calc':  probabilities calculated according to Born rule.
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            np.float: quantum probability.
        """
        if source.lower() == 'meas': 
            if self.meas_qcorr is not None:
                return self._meas_qcorr[self._check_int(row)]
            else:
                raise qiskit.QiskitError(
                    "To return measured quantum correlations please add qiskit results.")
        elif source.lower() == 'calc':
            return self._calc_qcorr.setdefault(row,self.quantum_prob(row))
        else:
            raise qiskit.QiskitError(
                "Invalid source of quantum correlations. Should be 'meas' or 'calc'.")

    @property
    def meas_qcorr(self) -> List[float]:
        """ Return a list of the measured quantum correlations with indices corresponding to the 
        LP rows.
            
        The setter validates if the sum of correlations is correct."""
        return self._meas_qcorr
    
    
    @meas_qcorr.setter
    def meas_qcorr(self, corr: List[float]):
        if not np.isclose(sum(corr),np.prod(self.s)): 
            raise qiskit.QiskitError('Invalid sum of correlations.')
        self._meas_qcorr = corr
                       
            
    @property
    def calc_qcorr(self) -> Dict[int, float]:
        """ Return a dict mapping row indices to the calculated quantum correlations.
        
        The setter validates the type of argument."""
        return self._calc_qcorr
    
    
    @calc_qcorr.setter
    def calc_qcorr(self, corr: Dict[int, np.float]):
        if not isinstance(corr,dict):
            raise qiskit.QiskitError(
                "corr should be a dict mapping rows to the corresponding quantum probabilities.")
        self._calc_qcorr = corr

        
    def quantum_prob(self, row: int) -> np.float:
        """ Return quantum probability according to Born rule."""
        if self.pre_meas_gates is None or self.rho is None:
            raise qiskit.QiskitError(
                "To calculate quantum probability, please provide rho and pre_meas_gates.")
        m = self.inputs(row)
        rid = int(float(row))%(self.d**self.n)
        U = functools.reduce(np.kron, [np.linalg.inv( self.pre_meas_gates[i][j] ) 
                                for i, j in enumerate(m)])
        p = 0*1j
        for l in range(self.d**self.n):
            for k in range(self.d**self.n):
                p += U[k][rid].conjugate() * self.rho[k][l] * U[l][rid]   
        return  np.real(p)
    

    # linear programming
    #-----------------------------------------------------------------------------------------------
    @property
    def LP(self) -> cplex.Cplex:
        """ Return the cplex LP model."""
        return self._LP
     
        
    def new_LP(self, LP_type: str) -> None:
        """ Build new cplex LP model and set it to self.LP. 
        Args:
            LP_type (str): type of an optimization problem, can be either 'feasibility' 
                or 'optimization'.
        """
        self._LP = cplex.Cplex()
        
        # set objective
        if LP_type.lower() == 'feasibility':
            self.LP.variables.add(ub = np.ones(self.cols), 
                                  lb = np.zeros(self.cols))
        elif LP_type.lower() in ['optimization','optimisation']:
            self.LP.variables.add(obj = np.hstack((np.zeros(self.cols),[1.0])), 
                                  ub = np.ones(self.cols+1), 
                                  lb = np.zeros(self.cols+1))
        else: 
            raise qiskit.QiskitError(
                "Invalid LP_type. Should be either 'feasibility' or "
                "'optimization'. Objective hasn't been set.")      
        self.LP.objective.set_sense(self.LP.objective.sense.maximize)
        
        # summation constraint
        self.LP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = range(self.cols), 
                                         val = np.ones(self.cols))], 
            rhs = [1.0], senses = 'E', range_values = [0.0], names = ['sum'])
        
        # restrict output
        self.LP.set_log_stream(None)
        self.LP.set_error_stream(None)
        self.LP.set_warning_stream(None)
        self.LP.set_results_stream(None)
        
        # solving method
        alg = self.LP.parameters.lpmethod.values
        self.LP.parameters.lpmethod.set(alg.dual)
        self.LP.parameters.threads.set(3)
        self.LP.parameters.parallel.set(1)
        
        
    def populate_LP(self, LP_type: str, source: str, rows: Sequence[int]) -> None:
        """ Adds constraints to cplex LP model.
        
        Args:
            LP_type (str): type of an optimization problem, can be either 'feasibility' 
                or 'optimization'.
            source (str): source of quantum correlations, can be either 'meas' or 'calc'.
            rows (Sequence[int]): indices of constrains in the LP problem.
        """
        if isinstance(rows, str):
            raise TypeError(
                "String sequence can be ambiguous in multisetting scenario.")  
        if not hasattr(rows, '__iter__'):
            rows = [self._check_int(rows)]
        
        if LP_type.lower() == 'feasibility':
            self.LP.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=self.get_local_corr(row),
                    val=np.ones(len(self.local_corr.get(row)))) for row in rows],
                rhs=[self.get_quantum_corr(source, row) for row in rows],
                senses=np.full(len(rows),'E'), range_values=np.zeros(len(rows)))

        elif LP_type.lower() in ['optimization', 'optimisation']:
            self.LP.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=self.get_local_corr(row) + [self.cols],
                    val=[1.0]*len(self.local_corr[row]) + [self.d**(-self.n) 
                        - self.get_quantum_corr(source, row)]) for row in rows],
                rhs=np.full(len(rows), self.d**(-self.n)),
                senses=np.full(len(rows),'E'), range_values=np.zeros(len(rows)))
        else:
            raise qiskit.QiskitError(
                "Invalid LP_type. Should be either 'feasibility' or 'optimization'. "
                "Constraints hasn't been set.") 
       
    
    # handler
    #-----------------------------------------------------------------------------------------------                   
    def __handle_args(self, source, rows, nosignaling, min_size, max_size, step):
        
        if not isinstance(source,str) or source.lower() not in ['meas','calc']:
            raise ValueError("Source should be either 'meas' or 'calc'.")       
        if not isinstance(nosignaling,(bool,np.bool_)):
            raise TypeError("nosignaling should be True or False.")        
            
        if rows is None:
            rows = list(range(self.rows_a))
        elif isinstance(rows, str):
            raise TypeError(
                "String sequence can be ambiguous in multisetting scenario.") 
        elif not hasattr(rows, '__iter__'):
            rows = [self._check_int(rows)]
        else:
            rows = list(set(rows))  # remove duplicates 
        
        if max_size is None:  # solve the full LP at the end
            max_size = min(self.rows_b, len(rows)) if nosignaling else len(rows)
        else: 
            max_size = self._check_int(max_size)
        
        subset = []
        if min_size is None: # solve the LP without iteration
            min_size = max_size  
        elif hasattr(min_size,'__iter__'): # start from custom initial subset
            if isinstance(min_size, str):
                raise TypeError(
                    "String sequence can be ambiguous in multisetting scenario.") 
            subset = list(set(min_size.copy())) # remove duplicates
            min_size = len(subset)
        else: 
            min_size = self._check_int(min_size)
        
        if not (1 <= min_size <= max_size <= len(rows)):
            raise ValueError("Invalid iterative bounds on subset size.")
 
        if step is None:
            step = ceil(len(rows)/100) 
        else:
            self._check_int(step)
        if not (1 <= step <= max_size):
            raise ValueError("Iterative step is out of range.") 
        
        return (rows, min_size, max_size, step, subset)
    
    
    #-----------------------------------------------------------------------------------------------
    def nonlocality_test(self, 
                         source: str = 'meas', 
                         rows: Optional[Sequence[int]] = None, 
                         nosignaling: bool = False,
                         min_size: Optional[Union[int, Sequence[int]]] = None, 
                         max_size: Optional[int] = None, 
                         step: Optional[int] = None, 
                         progressbar: bool = False): 
        """ Test Bell nonlocality.
        
        According to Bell theorem the correlations of some entangled states can't be described 
        by local realistic models. This method checks the feasibility of a classical model by 
        nonlocality was detected
        
        Args:
            source (str): source of quantum correlations. [Default: 'meas']
                'meas': probabilities from qiskit backend results.
                'calc': probabilities calculated according to Born rule.
            rows (Sequence[int]): indices of constraints taken into account in solving LP. 
                If None then takes all constraints. [Default: None]
            nosignaling: (bool): if True then only nonredundand rows according to default 
                nosignaling rule are passed to LP. [Default: False]
            min_size (int or list[int]): initial number of constraints in iterative solving. 
                If None then solves only LP with max_size of constraints without iteration. 
                If given as sequence then starts iteration from custom initial subset of constraints. 
                [Default: None]
            max_size (int): maximal number of constraints in iterative solving. 
                If None then in final step solves the full LP with all rows. [Default: None]
            step (int): the increase in size of subset of constraints in iterative solving. 
                If None then ceil(1% of all constraints). [Default: None]
            progressbar (bool): show tqdm progressbar if True. [Default: False]
            
        Returns: 
            bool: True if Bell nonlocality was detected, which is possible only for some entangled 
                quantum states. False if quantum correlations can be reproduced classically within 
                Local Hidden Variables (LHV) model.
        """
        
        (rows, min_size, max_size, step, subset) = self.__handle_args(
            source=source, rows=rows, nosignaling=nosignaling, 
            min_size=min_size, max_size=max_size, step=step)
        
        self.new_LP('feasibility')  
        
        subset_size = len(subset)
        if subset_size > 0: 
            for row in subset: 
                rows.remove(row)        
            self.populate_LP('feasibility', source, subset)
            
        if progressbar: 
            pbar = tqdm(initial=0, total=max_size)    
        for size in itertools.chain(range(min_size, max_size, step), [max_size]):
            
            new_rows = []
            
            if not nosignaling:
                for new in range(size - subset_size):
                    new_rows.append(rows.pop(np.random.randint(len(rows))))        
            else:
                while len(new_rows) < size - subset_size and rows:
                    new = rows.pop(np.random.randint(len(rows)))
                    if self.get_nonredundant(new): new_rows.append(new)
                        
            if new_rows:
                subset_size += len(new_rows)
                self.populate_LP('feasibility', source, new_rows)

            self.LP.solve() 

            status = self.LP.solution.get_status_string()
            NL = True if status=='infeasible' else (
                False if status=='optimal' else status)
            
            if progressbar:
                pbar.update(subset_size - pbar.n)
                pbar.set_description_str(desc=str(NL), refresh=True)
            if NL or not rows:  
                break

        return NL
    
    
    #-----------------------------------------------------------------------------------------------
    def nonlocality_strength(self, 
                             source: str = 'meas', 
                             rows: Optional[Sequence[int]] = None, 
                             nosignaling: bool = False,
                             min_size: Optional[Union[int, Sequence[int]]] = None, 
                             max_size: Optional[int] = None, 
                             step: Optional[int] = None, 
                             progressbar: bool = False,
                             plot: bool = False): 
        """ Calculate Bell nonlocality strength:
        
        The nonlocality strength is understood as resistance to noise and defined as the amount of 
        white noise admixture required to completely suppress the nonclassical character of the 
        original quantum correlations. The calculation of strength is based on visibility parameter 
        :math:`v` in: :math:`\rho(v)= v\rho +(1-v)\rho_{white noise}`. This method involves solving 
        an optimization problem (LP) in which the visibility is maximized until the set of linear 
        constraints can no longer be satisfied. That returns a critical visibility parameter, while 
        :math: `nonlocality strength = 1 - critical visibility`.
        
        Args:
            source (str): source of quantum correlations. [Default: 'meas']
                'meas': probabilities from qiskit backend results.
                'calc': probabilities calculated according to Born rule.
            rows (list[int]): indices of constraints taken into account in 
                solving LP. If None then takes all constraints. [Default: None]
            nosignaling: (bool): if True then only nonredundand rows according 
                to default nosignaling rule are passed to LP. [Default: False]
            min_size (int or list[int]): initial number of constraints in 
                iterative solving. If None then solves only LP with max_size 
                of constraints without iteration. If given as sequence then
                starts iteration from custom initial subset of constraints. 
                [Default: None]
            max_size (int): maximal number of constraints in iterative solving. 
                If None then in final step solves the full LP with all rows.
                [Default: None]
            step (int): the increase in size of subset of constraints 
                in iterative solving. If None then ceil(1% of all constraints).
                [Default: None]
            progressbar (bool): show tqdm progressbar if True. [Default: False]
            plot (bool): if True then plot nonlocality strength convergence vs 
                the size of constraints set. [Default: False]
            
        Returns: 
            np.float: the nonlocality strength. 
        """
               
        (rows, min_size, max_size, step, subset) = self.__handle_args(
            source=source, rows=rows, nosignaling=nosignaling, 
            min_size=min_size, max_size=max_size, step=step)
        

        self.new_LP('optimization')   
        
        subset_size = len(subset)
        if subset_size > 0:  
            for row in subset: 
                Rows.remove(row)        
            self.populate_LP('optimization', source, subset)
            
        strength_dict={}
        prec_dict = {}
        
        if progressbar: 
            pbar = tqdm(initial = 0, total = max_size)    
        for size in itertools.chain(range(min_size, max_size, step), [max_size]):
                
            new_rows = []
            
            if not nosignaling:
                for new in range(size - subset_size):
                    new_rows.append(Rows.pop(np.random.randint(len(rows))))        
            else:
                while len(new_rows) < size - subset_size and rows:
                    new = rows.pop(np.random.randint(len(rows)))
                    if self.get_nonredundant(new): 
                        new_rows.append(new)
                        
            if new_rows:
                subset_size += len(new_rows)
                self.populate_LP('optimization', source, new_rows)
                
            self.LP.solve() 
                
            cv = self.LP.solution.get_objective_value()
            prec = self.LP.solution.get_float_quality(
                self.LP.solution.quality_metric.max_primal_infeasibility)
            
            strength_dict[size] = 1 - cv
            prec_dict[size] = prec
            
            if progressbar:
                pbar.update(size - pbar.n)
                pbar.set_description_str(desc=str([1-cv, prec]), refresh=True)   
            if not rows: 
                break

        if plot:  
            fig, ax = plt.subplots(1,figsize=(12,8))
            for item in ([ax.title,ax.xaxis.label,ax.yaxis.label] + 
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)
            
            size = sorted(strength_dict.keys())
            STRENGTH = np.array([strength_dict[key] for key in size])
            PREC = np.array([prec_dict[key] for key in size])
            ax.plot(size, STRENGTH, lw=1, color='blue', marker='.')
            ax.fill_between(size, STRENGTH+PREC, STRENGTH-PREC, facecolor='blue', alpha=0.3)
            ax.set_xlabel('subset size')
            ax.set_ylabel('critical visibility')
            ax.grid()
            ticks = plt.yticks()
            if (ticks[0][1]-STRENGTH[-1]) < (ticks[0][-1]-STRENGTH[-1])/40: 
                i = 1
            else: 
                i = 0
            ticks[0][i] = STRENGTH[-1]
            plt.yticks(ticks[0][i:-1])
            fig.tight_layout() 

        return [1 - cv, prec]
    

    #-----------------------------------------------------------------------------------------------
    def nonlocality_sub_test(self, 
                             source: str = 'meas', 
                             rows: Optional[Sequence[int]] = None, 
                             nosignaling: bool = False,
                             min_size: Optional[int] = None, 
                             max_size: Optional[int] = None, 
                             step: Optional[int] = None, 
                             runs: int = 1,
                             progressbar: bool = False,
                             plot: bool = False):   
        """ Subsampling row constraints in testing Bell nonlocality :
        
        Args:
            source (str): source of quantum correlations. [Default: 'meas']
                'meas': probabilities from qiskit backend results.
                'calc': probabilities calculated according to Born rule.
            rows (list[int]): indices of constraints taken into account in solving LP. 
                If None then takes all constraints. [Default: None]
            nosignaling: (bool): if True then only nonredundand rows according to default 
                nosignaling rule are passed to LP. [Default: False]
            min_size (int or list[int]): initial number of constraints in iterative solving. 
                If None then solves only LP with max_size of constraints without iteration. 
                If given as sequence then starts iteration from custom initial subset of 
                constraints. [Default: None]
            max_size (int): maximal number of constraints in iterative solving. 
                If None then in final step solves the full LP with all rows.[Default: None]
            step (int): the increase in size of subset of constraints in iterative solving. 
                if None then ceil(1% of all constraints).[Default: None]
            runs (int): the number of repetitions of a single step, each time with new random 
                subset of constraints.
            progressbar (bool): show tqdm progressbar if True. [Default: False]
            plot (bool): show plot of nonlocality strength vs size of constraints subset. 
                [Default: False]
        """
        (rows, min_size, max_size, step, subset) = self.__handle_args(
            source=source, rows=rows, nosignaling=nosignaling, 
            min_size=min_size, max_size=max_size, step=step)
        
        NL = {}

        if progressbar: 
            pbar = tqdm(initial=0, total=max_size, desc='size')
        for size in itertools.chain(range(min_size, max_size, step), [max_size]):
            
            if nosignaling and len(rows) < size:
                print("Not enough nonredundant rows for subset of size: {}".format(size))
                break
            
            nl = collections.defaultdict(int)          
            
            if progressbar: 
                Runs = tqdm(range(runs), desc='runs', leave=False)
            else: 
                Runs = range(runs)
            for run in Runs:
                if not nosignaling: 
                    subset = np.random.choice(rows, size, replace=False)    
                else:
                    Rows = rows.copy()
                    lenR = len(Rows)
                    subset = []
                    while len(subset) < size and lenR > 0:
                        new = Rows.pop(np.random.randint(lenR))
                        if self.get_nonredundant(new): 
                            subset.append(new)
                        else: 
                            rows.remove(new)
                        lenR -= 1                    
                    if len(subset) < size: 
                        break
                
                self.new_LP('feasibility')
                self.populate_LP('feasibility',source,subset)
                self.LP.solve()
                status = self.LP.solution.status[self.LP.solution.get_status()]
                if status=='infeasible': 
                    nl[True] += 1
                elif status=='optimal': 
                    nl[False] += 1
                else: 
                    nl[status] += 1        
            NL[size] = dict(nl)
            
            if progressbar :
                pbar.update(size - pbar.n)
                pbar.set_description_str(refresh=True)
        
        if plot:         
            fig, ax = plt.subplots(1,figsize=(12,8))
            for item in ([ax.title,ax.xaxis.label,ax.yaxis.label] 
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)
            ax.plot(sorted(NL.keys()),[NL[k].get(True,0)/sum(NL[k].values()) 
                                       for k in sorted(NL.keys())],lw=1,marker='.')
            ax.set_title('Bell nonlocality test with subsampling',fontsize=18)
            ax.set_xlabel('subset size')
            ax.set_ylabel('success rate of nonlocality detection')
            ax.grid()
            fig.tight_layout()             
        
        return 
    
    
    #-----------------------------------------------------------------------------------------------
    def nonlocality_sub_strength(self, 
                                 source: str = 'meas', 
                                 rows: Optional[Sequence[int]] = None, 
                                 nosignaling: bool = False,
                                 min_size: Optional[int] = None, 
                                 max_size: Optional[int] = None, 
                                 step: Optional[int] = None, 
                                 runs: int = 1,
                                 progressbar: bool = False,
                                 plot: bool = False): 
        """ Subsampling constraints in nonlocality strength calculations :
        
        Args:
            source (str): source of quantum correlations. [Default: 'meas']
                'meas': probabilities from qiskit backend results.
                'calc': probabilities calculated according to Born rule.
            rows (list[int]): indices of constraints taken into account in solving LP. 
                If None then takes all constraints. [Default: None]
            nosignaling: (bool): if True then only nonredundand rows according to default 
                nosignaling rule are passed to LP. [Default: False]
            min_size (int or list[int]): initial number of constraints in iterative solving. 
                If None then solves only LP with max_size of constraints without iteration. 
                If given as sequence then starts iteration from custom initial subset of 
                constraints. [Default: None]
            max_size (int): maximal number of constraints in iterative solving. 
                If None then in final step solves the full LP with all rows.[Default: None]
            step (int): the increase in size of subset of constraints in iterative solving. 
                if None then ceil(1% of all constraints).[Default: None]
            runs (int): the number of repetitions of a single step, each time with new random 
                subset of constraints.
            progressbar (bool): show tqdm progressbar if True. [Default: False]
            plot (bool): show plot of nonlocality strength vs size of constraints subset. 
                [Default: False]
        """
        
        (rows, min_size, max_size, step, subset) = self.__handle_args(
            source=source, rows=rows, nosignaling=nosignaling, 
            min_size=min_size, max_size=max_size, step=step)


        STRENGTHstat = {}
        
        if progressbar: 
            pbar = tqdm(initial=0, total=max_size, desc='size')
        for size in itertools.chain(range(min_size, max_size, step), [max_size]):
                
            if nosignaling and len(rows) < size:
                print("Not enough nonredundant rows for subset of size: {}".format(size))
                break

            STRENGTH = []

            if progressbar: 
                Runs = tqdm(range(runs), desc='runs', leave=False)
            else: 
                Runs = range(runs)
            for run in Runs:    
                if not nosignaling: 
                    subset = np.random.choice(rows, size, replace=False)    
                else:
                    Rows = rows.copy()
                    lenR = len(Rows)
                    subset = []
                    while len(subset) < size and lenR > 0:
                        new = Rows.pop(np.random.randint(lenR))
                        if self.get_nonredundant(new): 
                            subset.append(new)
                        else: 
                            rows.remove(new)
                        lenR -= 1                    
                    if len(subset) < size: 
                        break
                            
                self.new_LP('optimization')
                self.populate_LP('optimization',source,subset)
                self.LP.solve()
                cv = self.LP.solution.get_objective_value()

                STRENGTH.append(1 - cv)


            STRENGTHstat[size] = {'mean':np.mean(STRENGTH),'stddev':np.std(STRENGTH),
                                  stat':len(STRENGTH)}
                    
            if progressbar:
                pbar.update(size - pbar.n)
                pbar.set_description_str(refresh=True)

        if plot:        
            fig, ax = plt.subplots(1,figsize=(12,8))
            for item in ([ax.title,ax.xaxis.label,ax.yaxis.label] 
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)

            STRENGTHmean = np.array([STRENGTHstat[k]['mean'] for k in STRENGTHstat])
            STRENGTHstd = np.array([STRENGTHstat[k]['stddev'] for k in STRENGTHstat])
            ax.plot(STRENGTHstat.keys(), STRENGTHmean, lw=1, color='blue', marker='.')
            ax.fill_between(STRENGTHstat.keys(), STRENGTHmean+STRENGTHstd, 
                            STRENGTHmean-STRENGTHstd, facecolor='blue', alpha=0.3)
            ax.set_xlabel('subset size')
            ax.set_ylabel('critical visibility')
            ax.grid()
            ticks = plt.yticks()
            if (ticks[0][1] - STRENGTHmean[-1]) < (ticks[0][-1] - STRENGTHmean[-1])/20: 
                i = 1
            else: 
                i = 0
            ticks[0][i] = STRENGTHmean[-1]
            plt.yticks(ticks[0][i:-1])
            fig.tight_layout() 
                        
        return fig.tight_layout() 
