""" BellScenario - a parent class for Bell Nonlocality testing 
    - contains properties deduced from setting scenario of Bell experiment """

import numpy as np
import logging, string


class BellScenario():
    """ General scenario of Bell experiment """
    #--------------------------------------------------------------------------------------------
    def __init__(self, sett, d = 2):
        """ initializes attributes in BellScenario
        args:
            sett: specifies the number of possible measurement settings per system.
            len(sett): specifies the number of systems.
            d: specifies the number of possible outcomes (d = 2 for qubits).    
        """ 
        
        self.s = sett
        self.d = d

        self._local_corr = {}
        self._nonredundant = {}
    
    #--------------------------------------------------------------------------------------------    
    # Bell scenario attributes
    # -------------------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        """ returns the number of systems (i.e. number of qubits) """
        return len(self.s)

    @property
    def s(self) -> list:
        """ returns the list with numbers of possible measurement settings per system """
        return self._s
    @s.setter
    def s(self, s):
        assert hasattr(s,'__iter__') and not isinstance(s,str) and len(s) > 0,' settings should be given as a sequence of numbers'
        sett = []
        for i in s: 
            try:
                if isinstance(i,int): assert not isinstance(i,(bool,np.bool_)), ' setting can\'t be bool'
                else:
                    assert float(i).is_integer(),' {} should be int'.format(i)
                    i = int(float(i))
                assert i > 0,' {} should be int > 0'.format(i) 
            except AssertionError as aerr: logging.error(aerr)
            except TypeError as terr: logging.error(terr)
            except ValueError as verr: logging.error(verr) 
            else:
                sett.append(i)        
        self._s = sett
        
    @property
    def d(self) -> int:
        """ returns the number of possible outcomes equal to local Hilbert space dimension (the same for each system) """
        return self._d
    @d.setter
    def d(self, d):
        assert not isinstance(d,(bool,np.bool_)) and float(d).is_integer() and float(d) > 1,' d should be int > 1'
        self._d = int(float(d))
    
    #--------------------------------------------------------------------------------------------    
    # problem sizes
    # ------------------------------------------------------------------------------------------- 
    @property
    def rowsA(self) -> int:
        """ returns the dimension of space of behaviours (i.e. number of All rows in LP problem) """
        return int(np.prod(self.s,dtype=np.float64)*(self.d**self.n))
        
    @property
    def rowsB(self) -> int:
        """ returns the dimension of Bell polytope (i.e. number of rows in reduced LP problem) """
        return int(np.prod(np.array(self.s)*(self.d-1)+1,dtype=np.float64) - 1)
    
    @property
    def cols(self) -> int:
        """ returns the length of joint probabilities in LHV model (i.e. number of columns in LP problem) """
        return self.d**sum(self.s) 
       
    #--------------------------------------------------------------------------------------------    
    # row and inputs & output mapping methods
    # -------------------------------------------------------------------------------------------   
    def inputs(self, row: int) -> list:
        """ returns a list of indices of choosen measurements for each observer in a given row 
        args:
            row (int): index of row in LP problem 
        """ 
        try:
            if isinstance(row,int): assert not isinstance(row,(bool,np.bool_)), ' row can\'t be bool'
            else:
                assert float(row).is_integer(),' row should be int'
                row = int(float(row))
            assert 0 <= row < self.rowsA,' row index {0} out of range for this problem: 0 <= row < {1}'.format(row,self.rowsA)
        except AssertionError as aerr: logging.error(aerr)
        except TypeError as terr: logging.error(terr)
        except ValueError as verr: logging.error(verr) 
        else:
            m = []
            mid = row//(self.d**self.n)
            spr = [np.prod(self.s[-i:]) for i in range(1,self.n)]
            for i in range(1,self.n):
                m.append(mid // spr[-i])
                mid = mid % spr[-i]
            m.append(mid)
            return list(map(int,m))
        
        
    def outputs(self, row: int) -> list:
        """ returns a list of outcomes for each observer in a given row 
        args:
            row (int): index of row in LP problem 
        """
        try:
            if isinstance(row,int): assert not isinstance(row,(bool,np.bool_)), ' row can\'t be bool'
            else:
                assert float(row).is_integer(),' row should be int'
                row = int(float(row))
            assert 0 <= row < self.rowsA,' row index {0} out of range for this problem: 0 <= row < {1}'.format(row,self.rowsA)
        except AssertionError as aerr: logging.error(aerr)
        except TypeError as terr: logging.error(terr)
        except ValueError as verr: logging.error(verr) 
        else:
            r = []
            rid = row%(self.d**self.n)
            while rid > 0:
                r.append(rid%self.d)
                rid = rid//self.d
            if self.n - len(r) > 0: 
                r += ([0] * (self.n-len(r)))
            r = [x for x in r[::-1]]
            return list(map(int,r))     
    

    def row(self, inputs: list, outputs: list) -> int:    
        """ returns a row index corresponding to choosen measurements (inputs) and results (outputs) 
        args:
            inputs: list of indices of choosen measurements for each observer in a given row
            outputs: list of outcomes for each observer in a given row 
        """
        try:
            assert hasattr(inputs,'__iter__') and len(inputs) == self.n, ' len(inputs) doesn\'t match n'
            assert all(isinstance(i,(int,np.int64)) and not isinstance(i,(bool, np.bool_)) 
                       and 0 <= i < self.s[j] for j,i in enumerate(inputs)),' inputs out of range'
            assert hasattr(outputs,'__iter__') and len(outputs) == self.n, ' len(outputs) does not match n'
            assert all(isinstance(i,(int,np.int64)) and not isinstance(i,(bool, np.bool_)) 
                       and 0 <= i < self.d for i in outputs),' outputs out of range'     
        except AssertionError as exception:
            logging.exception(exception,exc_info=False) 
        else:
            rid = int(''.join([(string.digits+string.ascii_lowercase)[i] for i in outputs]),base=self.d)
            mid = np.inner(inputs,[np.prod(self.s[i:]) for i in range(1,self.n)]+[1])
            return mid*self.d**self.n + rid    

    #--------------------------------------------------------------------------------------------    
    # local correlations - indices that multiplies joint prob.
    # -------------------------------------------------------------------------------------------

    @property
    def local_corr(self) -> dict:
        """ returns dictionary {row: non0ind(row)} corresponding to Bell local correlations """ 
        return self._local_corr
    @local_corr.setter
    def local_corr(self, corr: dict):
        """ setter of local Bell correlations dictionary from file """ 
        try: assert isinstance(corr,dict)
        except AssertionError: logging.error(' corr should be a dictionary {row: non0ind(row)}')
        else: self._local_corr = corr
            
    def get_local_corr(self, row: int) -> list:
        """ returns a list of indices of nonzero elements in a given row scenario
            from dict of Bell local correlations and updates it if needed """
        try: return self._local_corr.setdefault(row,self._non0ind(row))
        except TypeError as terr: logging.error(' TypeError: {}'.format(terr)) 
    
    def _non0ind(self, row: int) -> list:
        """ returns a list of indices of nonzero elements in a given row scenario """
        m = self.inputs(row)
        if m is None: return 
        r = self.outputs(row)
        if r is None: return 
        m0 = np.cumsum(self.s) - self.s + m
        ind = []
        x = [0]*sum(self.s)
        def addind(i):
            if i == sum(self.s):
                ind.append(int(''.join(map(str, x)),self.d))
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
    
    #--------------------------------------------------------------------------------------------    
    # nonredundant rows due to nosignaling rules
    # -------------------------------------------------------------------------------------------
    @property
    def nonredundant(self) -> dict:
        """ returns dictionary {row: bool} with True for nonredundant rows due to nosignaling rules
            and False otherwise (for redundant rows)""" 
        return self._nonredundant
    
    @nonredundant.setter
    def nonredundant(self, nonredundant: dict):
        try: assert isinstance(nonredundant,dict)
        except AssertionError: logging.error(' nonredundant should be a dictionary {row: isnonredundant(row)}')
        else: self._nonredundant = nonredundant
            
    def get_nonredundant(self, row: int) -> bool:
        """ returns True/False for key in nonredundant dict updating it if needed """
        try: return self._nonredundant.setdefault(row,self.isnonredundant(row))
        except TypeError as err: logging.error(' TypeError: {}'.format(err))
                
    def nonredundant_full_list(self) -> list:
        """ returns full list of nonredundant rows """
        return [ row for row in range(self.rowsA) if self.get_nonredundant(row) ]
            
    def isnonredundant(self, row: int) -> bool:
        """ returns True/False if row is nonredundant/redundant in a given row scenario """ 
        # there is no unique way to choose non redundant rows according to nosignaling; here:
        # a row is redundant if any of the observers obtained a result equal to (d−1) 
        # while measuring an observable with index higher than 0
        m = self.inputs(row)
        if m is None: return
        r = self.outputs(row)
        if r is None: return
        
        return False if int(row)==0 or any([m[i]>0 and r[i]==(self.d-1) for i in range(self.n)]) else True


