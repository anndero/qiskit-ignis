# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=cell-var-from-loop


"""
Measurement correction fitters.
"""
from typing import List, Union
import copy
import re
import numpy as np
from qiskit import QiskitError
from qiskit.result import Result
from .filters import MeasurementFilter, TensoredFilter
from ...verification.tomography import count_keys

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CompleteMeasFitter():
    """
    Measurement correction fitter for a full calibration
    """

    def __init__(self,
                 results: Union[Result, List[Result]],
                 state_labels: List[str],
                 qubit_list: List[int] = None,
                 circlabel: str = ''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`

        A wrapper for the tensored fitter

        Args:
            results: the results of running the measurement calibration
                circuits. If this is `None` the user will set a calibration
                matrix later.
            state_labels: list of calibration state labels
                returned from `measurement_calibration_circuits`.
                The output matrix will obey this ordering.
            qubit_list: List of the qubits (for reference and if the
                subset is needed). If `None`, the qubit_list will be
                created according to the length of state_labels[0].
            circlabel: if the qubits were labeled.
        """

        if qubit_list is None:
            qubit_list = range(len(state_labels[0]))
        self._qubit_list = qubit_list

        self._tens_fitt = TensoredMeasFitter(results,
                                             [qubit_list],
                                             [state_labels],
                                             circlabel)

    @property
    def cal_matrix(self):
        """Return cal_matrix."""
        return self._tens_fitt.cal_matrices[0]

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        """set cal_matrix."""
        self._tens_fitt.cal_matrices = [copy.deepcopy(new_cal_matrix)]

    @property
    def state_labels(self):
        """Return state_labels."""
        return self._tens_fitt.substate_labels_list[0]

    @property
    def qubit_list(self):
        """Return list of qubits."""
        return self._qubit_list

    @state_labels.setter
    def state_labels(self, new_state_labels):
        """Set state label."""
        self._tens_fitt.substate_labels_list[0] = new_state_labels

    @property
    def filter(self):
        """Return a measurement filter using the cal matrix."""
        return MeasurementFilter(self.cal_matrix, self.state_labels)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        """
        Add measurement calibration data

        Args:
            new_results: a single result or list of results
            rebuild_cal_matrix: rebuild the calibration matrix
        """

        self._tens_fitt.add_data(new_results, rebuild_cal_matrix)

    def subset_fitter(self, qubit_sublist=None):
        """
        Return a fitter object that is a subset of the qubits in the original
        list.

        Args:
            qubit_sublist: must be a subset of qubit_list

        Returns:
            A fitter than has the calibration for a subset of qubits

        """

        if self._tens_fitt.cal_matrices is None:
            raise QiskitError("Calibration matrix is not initialized")

        if qubit_sublist is None:
            raise QiskitError("Qubit sublist must be specified")

        for qb in qubit_sublist:
            if qb not in self._qubit_list:
                raise QiskitError("Qubit not in the original set of qubits")

        # build state labels
        new_state_labels = count_keys(len(qubit_sublist))

        # mapping between indices in the state_labels and the qubits in
        # the sublist
        qubit_sublist_ind = []
        for sqb in qubit_sublist:
            for qbind, qb in enumerate(self._qubit_list):
                if qb == sqb:
                    qubit_sublist_ind.append(qbind)

        # states in the full calibration which correspond
        # to the reduced labels
        q_q_mapping = []
        state_labels_reduced = []
        for label in self.state_labels:
            tmplabel = [label[l] for l in qubit_sublist_ind]
            state_labels_reduced.append(''.join(tmplabel))

        for sub_lab_ind, _ in enumerate(new_state_labels):
            q_q_mapping.append([])
            for labelind, label in enumerate(state_labels_reduced):
                if label == new_state_labels[sub_lab_ind]:
                    q_q_mapping[-1].append(labelind)

        new_fitter = CompleteMeasFitter(results=None,
                                        state_labels=new_state_labels,
                                        qubit_list=qubit_sublist)

        new_cal_matrix = np.zeros([len(new_state_labels),
                                   len(new_state_labels)])

        # do a partial trace
        for i in range(len(new_state_labels)):
            for j in range(len(new_state_labels)):

                for l in q_q_mapping[i]:
                    for k in q_q_mapping[j]:
                        new_cal_matrix[i, j] += self.cal_matrix[l, k]

                new_cal_matrix[i, j] /= len(q_q_mapping[i])

        new_fitter.cal_matrix = new_cal_matrix

        return new_fitter

    def readout_fidelity(self, label_list=None):
        """
        Based on the results, output the readout fidelity which is the
        normalized trace of the calibration matrix

        Args:
            label_list: If `None`, returns the average assignment fidelity
                of a single state. Otherwise it returns the assignment fidelity
                to be in any one of these states averaged over the second
                index.

        Returns:
            readout fidelity (assignment fidelity)

        Additional Information:
            The on-diagonal elements of the calibration matrix are the
            probabilities of measuring state 'x' given preparation of state
            'x' and so the normalized trace is the average assignment fidelity
        """
        return self._tens_fitt.readout_fidelity(0, label_list)

    def plot_calibration(self, ax=None, show_plot=True):
        """
        Plot the calibration matrix (2D color grid plot)

        Args:
            show_plot (bool): call plt.show()

        """

        self._tens_fitt.plot_calibration(0, ax, show_plot)


class TensoredMeasFitter():
    """
    Measurement correction fitter for a tensored calibration.
    """

    def __init__(self,
                 results: Union[Result, List[Result]],
                 mit_pattern: List[List[int]],
                 substate_labels_list: List[List[str]] = None,
                 circlabel: str = ''):
        """
        Initialize a measurement calibration matrix from the results of running
        the circuits returned by `measurement_calibration_circuits`.

        Args:
            results: the results of running the measurement calibration
                circuits. If this is `None`, the user will set calibration
                matrices later.

            mit_pattern: qubits to perform the
                measurement correction on, divided to groups according to
                tensors

            substate_labels_list: for each
                calibration matrix, the labels of its rows and columns.
                If `None`, the labels are ordered lexicographically

            circlable: if the qubits were labeled
        """

        self._result_list = []
        self._cal_matrices = None
        self._circlabel = circlabel

        self._qubit_list_sizes = \
            [len(qubit_list) for qubit_list in mit_pattern]

        self._indices_list = []
        if substate_labels_list is None:
            self._substate_labels_list = []
            for list_size in self._qubit_list_sizes:
                self._substate_labels_list.append(count_keys(list_size))
        else:
            self._substate_labels_list = substate_labels_list
            if len(self._qubit_list_sizes) != len(substate_labels_list):
                raise ValueError("mit_pattern does not match \
                    substate_labels_list")

        self._indices_list = []
        for _, sub_labels in enumerate(self._substate_labels_list):
            self._indices_list.append(
                {lab: ind for ind, lab in enumerate(sub_labels)})

        self.add_data(results)

    @property
    def cal_matrices(self):
        """Return cal_matrices."""
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        """Set _cal_matrices."""
        self._cal_matrices = copy.deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        """Return _substate_labels_list."""
        return self._substate_labels_list

    @property
    def filter(self):
        """Return a measurement filter using the cal matrices."""
        return TensoredFilter(self._cal_matrices, self._substate_labels_list)

    @property
    def nqubits(self):
        """Return _qubit_list_sizes."""
        return sum(self._qubit_list_sizes)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        """
        Add measurement calibration data

        Args:
            new_results: a single result or list of results
            rebuild_cal_matrix: rebuild the calibration matrix
        """

        if new_results is None:
            return

        if not isinstance(new_results, list):
            new_results = [new_results]

        for result in new_results:
            self._result_list.append(result)

        if rebuild_cal_matrix:
            self._build_calibration_matrices()

    def readout_fidelity(self, cal_index=0, label_list=None):
        """
        Based on the results, output the readout fidelity, which is the average
        of the diagonal entries in the calibration matrices.

        Args:
            cal_index(integer): readout fidelity for this index in _cal_matrices
            label_list (list of lists on states):
                Returns the average fidelity over of the groups of states.
                If `None`, then each state used in the construction of the
                calibration matrices forms a group of size 1

        Returns:
            readout fidelity (assignment fidelity)


        Additional Information:
            The on-diagonal elements of the calibration matrices are the
            probabilities of measuring state 'x' given preparation of state
            'x'.
        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if label_list is None:
            label_list = [[label] for label in
                          self._substate_labels_list[cal_index]]

        state_labels = self._substate_labels_list[cal_index]
        fidelity_label_list = []
        if label_list is None:
            fidelity_label_list = [[label] for label in state_labels]
        else:
            for fid_sublist in label_list:
                fidelity_label_list.append([])
                for fid_statelabl in fid_sublist:
                    for label_idx, label in enumerate(state_labels):
                        if fid_statelabl == label:
                            fidelity_label_list[-1].append(label_idx)
                            continue

        # fidelity_label_list is a 2D list of indices in the
        # cal_matrix, we find the assignment fidelity of each
        # row and average over the list
        assign_fid_list = []

        for fid_label_sublist in fidelity_label_list:
            assign_fid_list.append(0)
            for state_idx_i in fid_label_sublist:
                for state_idx_j in fid_label_sublist:
                    assign_fid_list[-1] += \
                        self._cal_matrices[cal_index][state_idx_i][state_idx_j]
            assign_fid_list[-1] /= len(fid_label_sublist)

        return np.mean(assign_fid_list)

    def _build_calibration_matrices(self):
        """
        Build the measurement calibration matrices from the results of running
        the circuits returned by `measurement_calibration`.
        """

        # initialize the set of empty calibration matrices
        self._cal_matrices = []
        for list_size in self._qubit_list_sizes:
            self._cal_matrices.append(np.zeros([2**list_size, 2**list_size],
                                               dtype=float))

        # go through for each calibration experiment
        for result in self._result_list:
            for experiment in result.results:
                circ_name = experiment.header.name
                # extract the state from the circuit name
                # this was the prepared state
                circ_search = re.search('(?<=' + self._circlabel + 'cal_)\\w+',
                                        circ_name)

                # this experiment is not one of the calcs so skip
                if circ_search is None:
                    continue

                state = circ_search.group(0)

                # get the counts from the result
                state_cnts = result.get_counts(circ_name)
                for measured_state, counts in state_cnts.items():
                    end_index = self.nqubits
                    for cal_ind, cal_mat in enumerate(self._cal_matrices):

                        start_index = end_index - \
                            self._qubit_list_sizes[cal_ind]

                        substate_index = self._indices_list[cal_ind][
                            state[start_index:end_index]]
                        measured_substate_index = \
                            self._indices_list[cal_ind][
                                measured_state[start_index:end_index]]
                        end_index = start_index

                        cal_mat[measured_substate_index][substate_index] += \
                            counts

        for mat_index, _ in enumerate(self._cal_matrices):
            sums_of_columns = np.sum(self._cal_matrices[mat_index], axis=0)
            # pylint: disable=assignment-from-no-return
            self._cal_matrices[mat_index] = np.divide(
                self._cal_matrices[mat_index], sums_of_columns,
                out=np.zeros_like(self._cal_matrices[mat_index]),
                where=sums_of_columns != 0)

    def plot_calibration(self, cal_index=0, ax=None, show_plot=True):
        """
        Plot one of the calibration matrices (2D color grid plot).

        Args:
            cal_index(integer): calibration matrix to plot
            ax(matplotlib.axes): settings for the graph
            show_plot (bool): call plt.show()

        Raises:
            QiskitError: if _cal_matrices was not set.

            ImportError: if matplotlib was not installed.

        """

        if self._cal_matrices is None:
            raise QiskitError("Cal matrix has not been set")

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        axim = ax.matshow(self.cal_matrices[cal_index],
                          cmap=plt.cm.binary,
                          clim=[0, 1])
        ax.figure.colorbar(axim)
        ax.set_xlabel('Prepared State')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Measured State')
        ax.set_xticks(np.arange(len(self._substate_labels_list[cal_index])))
        ax.set_yticks(np.arange(len(self._substate_labels_list[cal_index])))
        ax.set_xticklabels(self._substate_labels_list[cal_index])
        ax.set_yticklabels(self._substate_labels_list[cal_index])

        if show_plot:
            plt.show()
