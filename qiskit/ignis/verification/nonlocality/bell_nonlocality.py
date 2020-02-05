"""
Bell nonlocality test

test result = True means that nonlocality was detected - possible only for some entangled quantum states
test result = False means that correlations can be reproduced within LHV model
"""
from qiskit import *
import cplex
import numpy as np
import itertools as it
import functools as ft
import collections as coll
import matplotlib.pyplot as plt
#import scipy.optimize as scop

import sys
import logging
import tqdm
#from tqdm import tqdm_notebook as tqdm

from qiskit.ignis.verification.nonlocality.bell_scenario import BellScenario


class BLocFitter(BellScenario):
    """ class for Bell nonLocality tests """
    # --------------------------------------------------------------------------------------------

    def __init__(self, sett, backend_results=None, statevector=None, unitary_seq=None):
        """ initializes BellScenario, adds backend_results, statevector and unitary_seq
        Args:
            sett: specifies the number of possible measurement settings per system

            backend_results: list of backend results of sorted circuits
                             ( assumes the order of circuits as from meas_circs()! )

            statevector: statevector, needed to perform theorethical tests

            unitary_seq: sequence of unitary matrices which rotate measurement basis, 
                         assumes form as from meas_circs(), needed to perform theorethical tests
        """
        # initialize bell scenario
        super().__init__(sett=sett, d=2)

        # initialize internal variables

        self._meas_quant_corr = []  # measured quantum correlations based on backend_results
        # calculated quantum correlations based on rho and unitary_seq
        self._calc_quant_corr = {}

        self._results = []  # store all added backend_results as list
        self.add_results(backend_results)  # add new backend results to list

        self._rho = None
        self.rho = statevector  # density matrix build from a statevector
        self.u = unitary_seq

        self._LP = None

        self._LHVmodel = None
        self._minCV = None
        self._optimal_gates = None

        self._test_data = []  # [True/False, [used_constraints]]
        self._cv_data = {}  # [{size: [cv,prec,added_constraints]}]
        self._sub_test_data = {}  # {size: {True: #, False: # }}
        self._sub_cv_data = {}  # {size: [cv,cv,cv,...]}

    # --------------------------------------------------------------------------------------------
    # qiskit backend results --> measured quantum correlations
    # -------------------------------------------------------------------------------------------
    @property
    def results(self):
        """ returns a list of all uploaded backend results """
        return self._results

    def add_results(self, backend_results, clear=False):
        """
        Adds backend results. Updates measured quantum correlations
        Args:
            backend_results: list of backend results of sorted circuits
            clear (bool): if True then clears all previously added results 
        Additional information:
            assumes results come from executing circuits in order as from meas_circs() 
            number of shots in all results in the list should be the same
        """
        try:
            assert len(backend_results) == np.prod(self.s), \
                ' backend results should be a list of length {}'.format(
                    np.prod(self.s))
            assert all(isinstance(result, qiskit.result.result.Result) for result in backend_results), \
                ' backend results should be a list of qiskit result objects'
            assert len(set([result.to_dict()['results'][0]['shots'] for result in backend_results])) == 1, \
                ' number of shots in all results in backend_result list should be the same'
        except AssertionError as aerr:
            if backend_results == []:
                return
            else:
                logging.error(aerr)
        else:
            if clear:
                self._results = []
            self._results.append(backend_results)

            # calculate measured quantum correlations
            total_counts = np.zeros(self.rowsA)
            shots = 0
            for results in self._results:
                shots += results[0].to_dict()['results'][0]['shots']
                counts = []
                for result in results:
                    counts.extend(
                        [v for k, v in sorted(result.get_counts().items())])
                total_counts += np.array(counts)

            self._meas_quant_corr = total_counts/shots

    # --------------------------------------------------------------------------------------------
    # density matrix of quantum state
    # -------------------------------------------------------------------------------------------
    @property
    def rho(self):
        """ returns the density matrix of a state """
        return self._rho

    @rho.setter
    def rho(self, state):
        """ sets the density matrix of a state
        Args:
            state: can be either a statevector or a density matrix 
        """
        # from statevector
        if np.array(state).ndim == 1 and len(state) == self.d**self.n and np.isclose(sum(np.conj(i)*i for i in state), 1.0):
            self._rho = np.outer(state, np.conj(state)).astype('complex')
        # validate density matrix
        elif self._test_rho(state):
            self._rho = state.astype('complex')
        else:
            self._rho = None

    @staticmethod
    def _test_rho(rho):
        """ tests if a matrix is a correct density matrix """
        if np.array(rho).ndim != 2:
            return False  # matrix
        if len(set(rho.shape)) != 1:
            return False  # square matrix
        if not np.allclose(np.trace(rho), 1.0):
            return False  # trace = 1
        if not np.allclose(rho, np.transpose(np.conj(rho))):
            return False  # Hermitian
        # positive semidefinite
        return np.all(v >= 0 or np.isclose(v, 0.0, atol=1e-15) for v in np.linalg.eigvals(rho).real)

    # --------------------------------------------------------------------------------------------
    # unitary sequence
    # -------------------------------------------------------------------------------------------
    @property
    def u(self):
        """ returns sequence of unitary matrices which 'rotate' the measurement basis """
        return self._u

    @u.setter
    def u(self, u):
        """ setter checks lists lengths, but assumes that lists' elements are unitary matrices;
            also clears """
        if len(u) == self.n and all(len(u[i]) == self.s[i] for i in range(self.n)):
            self._u = u
        else:
            self._u = None
        self._calc_quant_corr.clear()  # clear calculated

    # --------------------------------------------------------------------------------------------
    # quantum correlations
    # -------------------------------------------------------------------------------------------
    def get_quantum_corr(self, source: str, row: int) -> np.float64:
        """ returns quantum correlation 
        Args: 
            souce: 'meas' - to get correlation from qiskit backend
                   'calc' - to get quantum probability calculated from rho and unitaries 
            row: index of a row in LP problem
        """

        if source.lower() == 'meas':
            return self._meas_quant_corr[row]
        elif source.lower() == 'calc':
            return self._calc_quant_corr.setdefault(row, self.quantum_prob(row))
        else:
            logging.error(
                ' source of quantum correlations not recognized, can be \'meas\' or \'calc\'')

    @property
    def meas_quant_corr(self) -> list:
        """ returns the list of measured quantum correlations with indices corresponding to LP rows"""
        return self._meas_quant_corr

    @meas_quant_corr.setter
    def meas_quant_corr(self, corr: list):
        if np.isclose(sum(corr), np.prod(self.s)):
            self._meas_quant_corr = corr
        else:
            self._meas_quant_corr = []

    @property
    def calc_quant_corr(self) -> dict:
        """ returns dictionary {row: quantum_prob(row)} of already calculated quantum correlations """
        return self._calc_quant_corr

    @calc_quant_corr.setter
    def calc_quant_corr(self, corr: dict):
        try:
            assert isinstance(corr, dict)
        except AssertionError:
            logging.error(
                ' corr should be a dictionary {row: quantum_prob(row)}')
        else:
            self._calc_quant_corr = corr

    def quantum_prob(self, row: int) -> np.float64:
        """ quantum probability for given rho and set of observables in given row correlation """
        if self.u == None:
            logging.error(' self.u is not defined')
            return
        m = self.inputs(row)
        if m is None:
            return
        rid = row % (self.d**self.n)
        U = ft.reduce(np.kron, [self.u[i][j] for i, j in enumerate(m)])
        p = 0*1j
        for l in range(self.d**self.n):
            for k in range(self.d**self.n):
                p += U[k][rid].conjugate() * self.rho[k][l] * U[l][rid]
        return np.real(p)

    # --------------------------------------------------------------------------------------------
    # linear programming
    # -------------------------------------------------------------------------------------------
    @property
    def LP(self):
        """ returns the cplex LP model """
        return self._LP

    def new_LP(self, LP_type):
        """ builds new cplex LP model """
        self._LP = cplex.Cplex()

        # set objective
        if LP_type.lower() == 'feasibility':
            self.LP.variables.add(ub=np.ones(self.cols),
                                  lb=np.zeros(self.cols))
        elif LP_type.lower() in ['optimization', 'optimisation']:
            self.LP.variables.add(obj=np.concatenate((np.zeros(self.cols), [1.0])),
                                  ub=np.ones(self.cols+1), lb=np.zeros(self.cols+1))
        else:
            logging.error(
                ' Invalid LP_type. Should be \'feasibility\' or \'optimization\'. Objective hasn\'t been set.')
        self.LP.objective.set_sense(self.LP.objective.sense.maximize)

        # summation constraint
        self.LP.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=range(self.cols), val=np.ones(self.cols))],
            rhs=[1.0], senses='E', range_values=[0.0], names=['sum'])

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

    def populate_LP(self, rows, LP_type, source):
        """ adds constraints to LP """
        try:
            if isinstance(rows, (int, np.int64)) and not(isinstance(rows, (bool, np.bool_))):
                rows = [rows]
            assert hasattr(rows, '__iter__'), ' wrong rows'
        except AssertionError as exception:
            logging.exception(exception, exc_info=False)
            return

        #assert source.lower() in ['meas','calc']
        try:
            if LP_type.lower() == 'feasibility':

                self.LP.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=self.get_local_corr(row),
                                               val=np.ones(len(self.local_corr.get(row)))) for row in rows],
                    rhs=[self.get_quantum_corr(source, row) for row in rows],
                    senses=np.full(len(rows), 'E'), range_values=np.zeros(len(rows)))

            elif LP_type.lower() in ['optimization', 'optimisation']:

                self.LP.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=self.get_local_corr(row) + [self.cols],
                        val=[1.0]*len(self.local_corr[row]) + [self.d**(-self.n) - self.get_quantum_corr(source, row)]) for row in rows],
                    rhs=np.full(len(rows), self.d**(-self.n)),
                    senses=np.full(len(rows), 'E'), range_values=np.zeros(len(rows)))

            else:
                logging.error(
                    ' Invalid LP_type. Constraints hasn\'t been added.')
        except TypeError:
            logging.error(' Invalid rows. Constraints hasn\'t been added.')

    # --------------------------------------------------------------------------------------------

    def test(self, source='meas', rows=None, nosignaling=False,  # basic args to build LP
             min_size=None, max_size=None, step=None,  # args for iterative solving
             progressbar=False):
        """ Tests Bell nonlocality :

            source: source of quantum correlations
                  'meas':  probabilities from qiskit backend results
                  'calc':  probabilities calculated according to Born rule for given state and measurements

            rows: indices of constraints (rows/correlations) taken into account in solving LP:
                iterable:  rows should be given as a sequence (e.g. list) of numbers
                    None:  all constraints as default

            nosignaling: (bool): if True than only non redundand rows according to default nosignaling rule 
                         are passed to LP; for small problems it's more efficient to call test with 
                         predetermined nonredundant rows as rows than checking redundancy of random rows)

            min_size: 
                     int:  initial number of constraints in iterative solving 
                    None:  min_size = max_size (no iteration, solves only LP with max_size of constraints)
                sequence:  min_size can be also provided as initial subset of constraints 

            max_size: 
                     int:  maximal number of constraints in iterative solving 
                    None:  max_size = len(rows) (in final step solves the full LP)

            step: (int):  the increase in size of subset of constraints in iterative solving

            progressbar: (bool):  show tqdm progressbar if True

        Returns: True / False
        """
        try:
            assert source.lower() in [
                'meas', 'calc'], ' source should either \'meas\' or \'calc\''
            assert nosignaling in [
                True, False], ' nosignaling should be True or False'

            if rows == None:
                Rows = range(self.rowsA)
            else:
                assert hasattr(rows, '__iter__') and not(isinstance(
                    rows, str)), ' rows should be iterable and not str'
                Rows = set(rows)  # remove duplicates
            assert len(
                Rows) < sys.maxsize, ' problem is too big - will be handle differently'
            Rows = list(Rows)

            if not max_size:  # solve the full LP at the end
                if nosignaling:
                    # full LP can be smaller than max_size because of redundancy checking
                    max_size = min(self.rowsB, len(Rows))
                else:
                    max_size = len(Rows)
            else:
                assert float(max_size).is_integer() and not isinstance(
                    max_size, (bool, np.bool_)), ' max_size should be int'

            subset = []
            if min_size == None:
                min_size = max_size  # solve the LP without iteration
            # custom initial subset
            elif hasattr(min_size, '__iter__') and not(isinstance(min_size, str)):
                subset = list(set(min_size.copy()))
                min_size = len(subset)
            else:
                assert int(min_size) == min_size and not isinstance(
                    min_size, (bool, np.bool_)), ' min_size should be int'

            assert 1 <= min_size <= max_size <= len(
                Rows), ' wroung bounds on subset size'

            if step == None:
                step = 1
            assert int(step) == step and not isinstance(
                step, (bool, np.bool_)), ' step should be int'
            assert 1 <= step <= max_size, ' step out of range'

        except AssertionError as aerr:
            logging.exception(aerr, exc_info=False)
            return

        self.new_LP('feasibility')

        subset_size = len(subset)
        if subset_size > 0:
            for row in subset:
                Rows.remove(row)
            self.populate_LP(subset, 'feasibility', source)

        if progressbar:
            pbar = tqdm.tqdm(initial=0, total=max_size)
        for size in it.chain(range(min_size, max_size, step), [max_size]):

            new_rows = []

            if not nosignaling:
                for new in range(size - subset_size):
                    new_rows.append(Rows.pop(np.random.randint(len(Rows))))
            else:
                while len(new_rows) < size - subset_size and Rows:
                    new = Rows.pop(np.random.randint(len(Rows)))
                    if self.nonredundant_get(new):
                        new_rows.append(new)

            if new_rows != []:
                subset_size += len(new_rows)
                self.populate_LP(new_rows, 'feasibility', source)

            self.LP.solve()

            status = self.LP.solution.get_status_string()
            NL = True if status == 'infeasible' else (
                False if status == 'optimal' else status)
            if progressbar:
                pbar.update(subset_size - pbar.n)
                pbar.set_description_str(desc=str(NL), refresh=True)
            if NL:
                break
            if not Rows:
                break

        return NL
