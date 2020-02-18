""" 
BellScenario - a parent class for Bell Nonlocality testing.
"""

import numpy as np
import string


class BellScenario():
    """ General scenario of Bell experiment. 

    Describes the general scenario of Bell experiment characterized by (n,s,d).
    Relates to optimization problems in testing Bell nonlocality:
    :class: 'qiskit.ignis.verification.nonlocality.bell_nonlocality'. 
    Contains properties and methods that can be deduced solely from Bell 
    scenario prior to employment of quantum correlations, i.e.:
        * sizes of optimization problem,
        * mapping between index of constraint and inputs, outputs,
        * nonzero elements in local model of correlations,
        * verification of constraints' redundancy. 

    Attributes:
        n (int): the number of systems (i.e. number of qubits).
        s (list[int]): possible measurement settings per system.
        d (int): the number of possible outcomes, equal to local 
            Hilbert space dimension (the same for each system).
        rowsA (int): the dimension of space of behaviours 
            (i.e. number of All rows in LP problem).
        rowsB (int): the dimension of Bell polytope 
            (i.e. number of rows in reduced LP problem).
        cols (int): the length of joint probabilities in LHV model 
            (i.e. number of columns in LP problem).
        local_corr (dict): the dict mapping row indices to nonzero elements 
            in Bell's local model of correlations.
        nonredundant (dict): the dict mapping row indices to bool indicating 
            if a given row is nonredundant (True) or redundant (False).
            
    """

    def __init__(self, sett, d = 2):
        """ Initializes BellScenario with attributes.
        Args:
            sett: sequence that specifies the number of possible measurement 
                settings per system; len(sett) specifies the number of systems.
            d: specifies the number of possible outcomes [Default: 2 (qubits)].
        """
        self._s = []
        self.s = sett

        self._d = 2 
        self.d = d

        self._local_corr = {}
        self._nonredundant = {}

    
    # Bell scenario attributes
    # --------------------------------------------------------------------------
    @property
    def n(self) -> int:
        """ Returns the number of systems (i.e. number of qubits). """
        return len(self.s)

    @property
    def s(self) -> list:
        """ Returns possible measurement settings per system. """
        return self._s

    @s.setter
    def s(self, s):
        if isinstance(s, str):
            raise TypeError(
                'String sequence can be ambiguous in multisetting scenarios.') 
        sett = [self._check_int(i,'setting') for i in s]
        if any(i <= 0 for i in sett):
            raise ValueError('{} should be int > 0'.format(i))
        self._s = sett

    @property
    def d(self) -> int:
        """ Returns the number of possible outcomes, equal to 
            local Hilbert space dimension and the same for each system. """
        return self._d

    @d.setter
    def d(self, d):
        d = self._check_int(d,'d')
        if d <= 1:
            raise ValueError(' d should be > 1')
        self._d = d


    # problem sizes
    # --------------------------------------------------------------------------
    @property
    def rowsA(self) -> int:
        """ Returns the dimension of space of behaviours 
            (i.e. number of All rows in LP problem). """
        if self.s:
            return int(np.prod(self.s, dtype=np.float64)*(self.d**self.n)) 
        else:
            return 0

    @property
    def rowsB(self) -> int:
        """ Returns the dimension of Bell polytope 
            (i.e. number of rows in reduced LP problem). """
        return int(np.prod(np.array(self.s)*(self.d-1)+1, dtype=np.float64) - 1)

    @property
    def cols(self) -> int:
        """ Returns the length of joint probabilities in LHV model 
            (i.e. number of columns in LP problem). """
        return self.d**sum(self.s) if self.s else 0


    # row and inputs & output mapping methods
    # --------------------------------------------------------------------------
    def inputs(self, row: int) -> list:
        """ Returns the list of measurements' indices choosen by all observers. 
        Args:
            row (int): an index of a constraint in an optimization problem.
        Returns:
            list[int]: input indices corresponding to a given row.
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row,'row')
        if not (0 <= row < self.rowsA):
            raise ValueError('Row index {0} out of range for this problem: '
                             '0 <= row < {1}'.format(row, self.rowsA))
        m = []
        mid = row//(self.d**self.n)
        spr = [np.prod(self.s[-i:]) for i in range(1, self.n)]
        for i in range(1, self.n):
            m.append(mid // spr[-i])
            mid = mid % spr[-i]
        m.append(mid)
        return list(map(int, m))

    def outputs(self, row: int) -> list:
        """ Returns the list of outcomes received by all observers. 
        Args:
            row (int): an index of a constraint in an optimization problem.
        Returns:
            list[int]: output indices corresponding to a given row.
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row,'row')
        if not (0 <= row < self.rowsA):
            raise ValueError('Row index {0} out of range for this problem: '
                             '0 <= row < {1}'.format(row, self.rowsA))
        r = []
        rid = row % (self.d**self.n)
        while rid > 0:
            r.append(rid % self.d)
            rid = rid//self.d
        if self.n - len(r) > 0:
            r += ([0] * (self.n-len(r)))
        r = [x for x in r[::-1]]
        return list(map(int, r))

    def row(self, inputs: list, outputs: list) -> int:
        """ Returns an index of constraint (row) in an optimization problem, 
            corresponding to choosen measurements (inputs) and results (outputs) 
        Args:
            inputs (list[int]): list of indices of choosen measurements.
            outputs (list[int]): list of outcomes for each observer.
        Returns:
            int: an index of a constraint in an optimization problem.
        Raises:
            ValueError: if either inputs or outputs are invalid.
        """
        if len(inputs) != self.n:
            raise ValueError('len(inputs) doesn\'t match n')
        inputs = [self._check_int(i,'input') for i in inputs]
        if not all( 0 <= i < self.s[j] for j, i in enumerate(inputs)):
            raise ValueError('some inputs out of range')
        
        if len(outputs) != self.n:
            raise ValueError('len(outputs) doesn\'t match n')
        outputs = [self._check_int(i,'output') for i in outputs]
        if not all( 0 <= i < self.d for i in outputs):
            raise ValueError('some outputs out of range')
        
        abc = string.digits+string.ascii_lowercase
        rid = int(''.join([abc[i] for i in outputs]), base=self.d)
        mid = np.inner(inputs,
                       [np.prod(self.s[i:]) for i in range(1, self.n)] + [1])
        return mid*self.d**self.n + rid


    # local correlations - indices that multiplies joint prob.
    # --------------------------------------------------------------------------
    @property
    def local_corr(self) -> dict:
        """ Returns the dictionary mapping row indices to lists of nonzero 
            elements in Bell's local model of correlations. """
        return self._local_corr

    @local_corr.setter
    def local_corr(self, corr: dict):
        if not isinstance(corr, dict):
            raise TypeError('corr should be dict')
        if not all(isinstance(key,int) for key in corr):
            raise ValueError('corr keys should be int')
        if not all(isinstance(val,list) for val in corr.values()):
            raise ValueError('corr values should be list')
        self._local_corr = corr

    def get_local_corr(self, row: int) -> list:
        """ Returns list of indices of nonzero elements in Bell's local model 
            of correlations for a given row. 
        Args:
            row (int): an index of a constraint in an optimization problem.
        Returns:
            list[int]: indices of nonzero elements (from self.local_corr dict 
                or result of self._non0ind(row) if a key is missing in dict).
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row,'row')
        return self._local_corr.setdefault(row, self._non0ind(row))

    def _non0ind(self, row: int) -> list:
        """ Determines indices of nonzero elements in Bell's local model of 
            correlations for a given row. 
        Args:
            row (int): an index of a constraint in an optimization problem.
        Returns:
            list[int]: indices of nonzero elements.
        """
        m = self.inputs(row)
        r = self.outputs(row)
        m0 = np.cumsum(self.s) - self.s + m
        ind = []
        x = [0]*sum(self.s)

        def addind(i):
            if i == sum(self.s):
                ind.append(int(''.join(map(str, x)), self.d))
                return ind
            elif i in m0:
                j = list(m0).index(i)
                x[i] = r[j]
                addind(i+1)
            else:
                for d0 in range(self.d):
                    x[i] = d0
                    addind(i+1)
        addind(0)
        return ind


    # nonredundant rows due to nosignaling rules
    # --------------------------------------------------------------------------
    @property
    def nonredundant(self) -> dict:
        """ Returns a dict mapping row indices to bool indicating if a given row 
            is nonredundant (True) or redundant (False) due to nosignaling rule. 
        """
        return self._nonredundant

    @nonredundant.setter
    def nonredundant(self, nonredundant: dict):
        if not isinstance(nonredundant, dict):
            return TypeError('nonredundant should be dict')
        if not all(isinstance(key,int) for key in nonredundant):
            raise ValueError('nonredundant keys should be int')
        if not all(isinstance(val,bool) for val in nonredundant.values()):
            raise ValueError('nonredundant values should be bool')
        self._nonredundant = nonredundant

    def get_nonredundant(self, row: int) -> bool:
        """ Returns a bool indicating if a given row is nonredundant (True) 
            or redundant (False) due to nosignaling rule. 
        Args:
            row (int): an index of a constraint in an optimization problem.
        Returns:
            bool: True if row is nonredundant, False otherwise 
                (from self.nonredundant dict or result of 
                self.isnonredundant(row) if a key is missing in dict).
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row,'row')
        return self._nonredundant.setdefault(row, self.isnonredundant(row))

    def nonredundant_full_list(self) -> list:
        """ Returns full list of nonredundant constraints. 
        Returns:
            list[int]: indices of nonredundant constraints."""
        return [row for row in range(self.rowsA) if self.get_nonredundant(row)]

    def isnonredundant(self, row: int) -> bool:
        """ Verifies redundancy of constraint in an optimization problem. 
        Args:
            row (int): an index of a constraint in an optimization problem.
        Returns:
            bool: True if row is nonredundant, False otherwise.
        Additional Information:
            There's no unique way to choose nonredundant rows due to nosignaling 
            Here: a row is redundant if any of the observers obtained a result 
            equal to (d−1) while measuring an observable with index > 0.
        """ 
        if self._check_int(row,'row') == 0:
            return False
        m = self.inputs(row)
        r = self.outputs(row)
        if any([m[i] > 0 and r[i] == (self.d-1) for i in range(self.n)]):
            return False
        else:
            return True
        
        
    # --------------------------------------------------------------------------
    @staticmethod 
    def _check_int(i, name='') -> int:
        """ Validates integer input.
        Args:
            i (int or float or complex or str): variable expected to be int.
            name (str): name of variable in an error message [Default: ''].
        Returns:
            int: arg i as int. 
        Raises:
            ValueError: if i can't be unambiguously interpret as integer.     
        """ 
        if isinstance (i, int):
            if isinstance(i, (bool, np.bool_)):
                raise ValueError(name +' can\'t be bool')
            else:
                return i
        if isinstance (i, complex):
            if i.imag == 0:
                i = i.real
            else:
                raise ValueError(name + ' should be real')
        if not float(i).is_integer(): 
            raise ValueError(name + ' should be int')    
        else:
            return int(float(i))            
