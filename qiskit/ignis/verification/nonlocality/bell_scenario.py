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

""" BellScenario - a parent class for Bell Nonlocality testing."""

from typing import List, Dict, Sequence
import string
import numpy as np


class BellScenario:
    #-----------------------------------------------------------------------------------------------
    """ General scenario of a Bell experiment. 
    
    Describes the general scenario of Bell experiment characterized by (n,s,d).
    Relates to optimization problems in testing Bell nonlocality:
    :class: 'qiskit.ignis.verification.nonlocality.bell_nonlocality'. 
    Contains properties and methods that can be deduced solely from Bell scenario prior to 
    employment of quantum correlations, i.e.:
        * sizes of optimization problem,
        * mapping between index of constraint and inputs, outputs,
        * nonzero elements in local model of correlations,
        * verification of constraints' redundancy.   
    """

    def __init__(self, sett: Sequence[int], d: int = 2):
        """ Initialize BellScenario with attributes.
        
        Args:
            sett (Sequence[int]): the number of possible measurement settings per subsystem. 
                len(sett) specifies the number of subsystems.
            d (int): specifies the number of possible outcomes [Default: 2 (qubits)].
        """
        self._s = []
        self.s = sett

        self._d = 2 
        self.d = d

        self._local_corr = {}
        self._nonredundant = {}

    
    # Bell scenario attributes
    #-----------------------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        """ Return the number of subsystems (i.e. number of qubits)."""
        return len(self.s)

    
    @property
    def s(self) -> List[int]:
        """ Return the number of possible measurement settings per subsystem.
        
        The setter validates if list contains only int > 0."""
        return self._s

    
    @s.setter
    def s(self, s: Sequence[int]):
        sett = [self._check_int(i) for i in s]
        if any(i <= 0 for i in sett):
            raise ValueError('setting should be int > 0.')
        self._s = sett

        
    @property
    def d(self) -> int:
        """ Return the number of possible outcomes, equal to the local Hilbert space dimension
        and the same for each subsystem."""
        return self._d

    
    @d.setter
    def d(self, d: int):
        d = self._check_int(d)
        if d <= 1:
            raise ValueError(' d should be > 1')
        self._d = d


    # problem sizes
    #-----------------------------------------------------------------------------------------------
    @property
    def rows_a(self) -> int:
        """ Return the dimension of the space of behaviours (i.e. number of All rows in LP 
        problem)."""
        return int(np.prod(self.s, dtype=np.float64)*(self.d**self.n)) if self.s else 0


    @property
    def rows_b(self) -> int:
        """ Return the dimension of Bell polytope (i.e. number of rows in reduced LP problem)."""
        return int(np.prod(np.array(self.s)*(self.d-1)+1, dtype=np.float64) - 1)
    

    @property
    def cols(self) -> int:
        """ Return the length of joint probabilities in LHV model (i.e. number of columns in LP 
        problem)."""
        return self.d**sum(self.s) if self.s else 0


    # row and inputs & output mapping methods
    #-----------------------------------------------------------------------------------------------
    def inputs(self, row: int) -> List[int]:
        """ Return the list of measurements' indices choosen by all observers. 
        
        Args:
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            List[int]: input indices corresponding to a given row.
            
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row)
        if not 0 <= row < self.rows_a:
            raise ValueError("Row index {0} out of range for this problem: 0 <= row < {1}".format(
                row, self.rows_a))
        m = []
        mid = row//(self.d**self.n)
        spr = [np.prod(self.s[-i:]) for i in range(1, self.n)]
        for i in range(1, self.n):
            m.append(mid // spr[-i])
            mid = mid % spr[-i]
        m.append(mid)
        return list(map(int, m))

    
    def outputs(self, row: int) -> List[int]:
        """ Return the list of outcomes received by all observers. 
        
        Args:
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            List[int]: output indices corresponding to a given row.
            
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row)
        if not 0 <= row < self.rows_a:
            raise ValueError("Row index {0} out of range for this problem: 0 <= row < {1}".format(
                row, self.rows_a))
        r = []
        rid = row % (self.d**self.n)
        while rid > 0:
            r.append(rid % self.d)
            rid = rid//self.d
        if self.n - len(r) > 0:
            r += ([0] * (self.n-len(r)))
        r = [x for x in r[::-1]]
        return list(map(int, r))

    
    def row(self, inputs: List[int], outputs: List[int]) -> int:
        """ Return an index of a constraint (row) in an optimization problem, corresponding to 
        choosen measurements (inputs) and results (outputs). 
        
        Args:
            inputs (List[int]): list of indices of choosen measurements.
            outputs (List[int]): list of outcomes for each observer.
            
        Returns:
            int: an index of a constraint in an optimization problem.
            
        Raises:
            ValueError: if either inputs or outputs are invalid.
        """
        if len(inputs) != self.n:
            raise ValueError("len(inputs) doesn't match n.")
        inputs = [self._check_int(i) for i in inputs]
        if not all(0 <= i < self.s[j] for j, i in enumerate(inputs)):
            raise ValueError("Some inputs are out of range.")
        
        if len(outputs) != self.n:
            raise ValueError("len(outputs) doesn't match n.")
        outputs = [self._check_int(i) for i in outputs]
        if not all(0 <= i < self.d for i in outputs):
            raise ValueError("Some outputs are out of range.")
        
        rid = int(''.join([string.digits+string.ascii_lowercase[i] for i in outputs]), base=self.d)
        mid = np.inner(inputs, [np.prod(self.s[i:]) for i in range(1, self.n)] + [1])
        return mid*self.d**self.n + rid


    # local correlations - indices that multiplies joint prob.
    #-----------------------------------------------------------------------------------------------
    @property
    def local_corr(self) -> Dict[int, List[int]]:
        """ Return the dictionary mapping row indices to lists of nonzero elements in Bell's local 
        model of correlations.
        
        The setter validates type of argument and its' items.        
        """
        return self._local_corr

    
    @local_corr.setter
    def local_corr(self, corr: Dict[int, List[int]]):
        if not isinstance(corr, dict):
            raise TypeError("corr should be dict.")
        if not all(isinstance(key, int) for key in corr):
            raise ValueError("corr keys should be int.")
        if not all(isinstance(val, list) for val in corr.values()):
            raise ValueError("corr values should be list.")
        self._local_corr = corr

        
    def get_local_corr(self, row: int) -> List[int]:
        """ Return a list of indices of nonzero elements in Bell's local model of correlations 
        for a given row. 
        
        Args:
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            List[int]: indices of nonzero elements (from self.local_corr dict or result of 
                self._non0ind(row) if a key is missing in dict).
                
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row)
        return self._local_corr.setdefault(row, self._non0ind(row))
    

    def _non0ind(self, row: int) -> List[int]:
        """ Determine indices of nonzero elements in Bell's local model of correlations for 
        a given row. 
        
        Args:
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            List[int]: indices of nonzero elements.
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
    #-----------------------------------------------------------------------------------------------
    @property
    def nonredundant(self) -> Dict[int, bool]:
        """ Return a dict mapping row indices to bool indicating if a given row is nonredundant 
        (True) or redundant (False) due to nosignaling rule. 
        
        The setter validates type of argument and its' items.
        """
        return self._nonredundant

    
    @nonredundant.setter
    def nonredundant(self, nonredundant: Dict[int, bool]):
        if not isinstance(nonredundant, dict):
            return TypeError("nonredundant should be dict.")
        if not all(isinstance(key, int) for key in nonredundant):
            raise ValueError("nonredundant keys should be int.")
        if not all(isinstance(val, bool) for val in nonredundant.values()):
            raise ValueError("nonredundant values should be bool.")
        self._nonredundant = nonredundant

        
    def get_nonredundant(self, row: int) -> bool:
        """ Return a bool indicating if a given row is nonredundant (True) or redundant (False) 
        due to nosignaling rule. 
        
        Args:
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            bool: True if row is nonredundant, False otherwise (from self.nonredundant dict or 
                result of self.isnonredundant(row) if a key is missing in dict).
                
        Raises:
            ValueError: if row isn't a valid row index.
        """
        row = self._check_int(row)
        return self._nonredundant.setdefault(row, self.isnonredundant(row))
    

    def nonredundant_full_list(self) -> List[int]:
        """ Return the full list of nonredundant constraints. 
        
        Returns:
            List[int]: indices of nonredundant constraints.
        """
        return [row for row in range(self.rows_a) if self.get_nonredundant(row)]
    

    def isnonredundant(self, row: int) -> bool:
        """ Verify redundancy of a constraint in an optimization problem. 
        
        There's no unique way to choose nonredundant rows due to nosignaling. 
        Here: a row is redundant if row = 0 or any of the observers obtained a result equal to 
        (d−1) while measuring an observable with index > 0.
        
        Args:
            row (int): an index of a constraint in an optimization problem.
            
        Returns:
            bool: True if row is nonredundant, False otherwise.
        """ 
        if self._check_int(row) == 0:
            return False
        m = self.inputs(row)
        r = self.outputs(row)
        return False if any([m[i] > 0 and r[i] == (self.d-1) for i in range(self.n)]) else True
        
        
    #-----------------------------------------------------------------------------------------------
    @staticmethod 
    def _check_int(i) -> int:
        if not isinstance(i, int) or isinstance(i, (bool, np.bool_)):
            raise ValueError("{} should be int and shouldn't be bool.".format(i))
        return i

    
    #@staticmethod 
    #def _check_int(i) -> int:
    #    """ Validate integer input.
    #    Args:
    #        i (int or float or complex or str): variable expected to be int.
    #    Returns:
    #        int: arg i as int. 
    #    Raises:
    #        ValueError: if i can't be unambiguously interpret as integer.     
    #    """ 
    #    if isinstance(i, int):
    #        if isinstance(i, (bool, np.bool_)):
    #            raise ValueError("Expected int but not bool.")
    #        else:
    #            return i
    #    if isinstance(i, complex):
    #        if i.imag == 0:
    #            i = i.real
    #        else:
    #            raise ValueError("Expected int should be real.")
    #    if not float(i).is_integer(): 
    #        raise ValueError("Expected should be int.")    
    #    else:
    #        return int(float(i))     
