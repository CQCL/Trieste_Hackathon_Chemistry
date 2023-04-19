import numpy as np

from pytket.circuit import Circuit
from pytket.utils import QubitPauliOperator

from scipy.linalg import expm

from pandas import DataFrame

import numpy

from enum import Enum
from typing import Union

class TimeEvolutionType(Enum):
    REAL = 0
    IMAG = 1


class TrotterTimeEvo:
    """This class Trotterises the time evolution operator using scipy matrix multiplication
    It is very fast but does not use circuits and therefor should just be used for testing and developing
    """

    @property
    def evolved_measuremets(self):
        return DataFrame.from_dict(self._evolved_measurements, orient='index',columns=[f'O_{i}' for i in range(len(self._measurements))]).rename_axis('Time').rename_axis('<O>', axis='columns')

    def __init__(
        self,
        initial_state: Union[Circuit,numpy.array],
        qubit_operator: QubitPauliOperator,
        measurements: list[QubitPauliOperator],
        t_max: float,
        n_trotter_steps: int,
        evolution_type: TimeEvolutionType,
        *args,
        **kwargs,
    ):

        self._n_qubits = initial_state.n_qubits
        if isinstance(initial_state,Circuit):
            self._initial_state = initial_state.get_statevector()
        else:
            self._initial_state = initial_state
        self._qubit_operator = qubit_operator.to_sparse_matrix(self._n_qubits)
        operator_paulis = [q.to_sparse_matrix(self._n_qubits) for q in qubit_operator.terms()]
        self._time_step = t_max / n_trotter_steps
        self._time_space = numpy.linspace(0, t_max, n_trotter_steps)
        self._measurements = [m.to_sparse_matrix(self._n_qubits) for m in measurements]
        self._evolution_type = evolution_type

        if evolution_type == TimeEvolutionType.IMAG:
            self._trotter_step = expm(-1 * self._qubit_operator * self._time_step)
        elif TimeEvolutionType.REAL:
            # Change this to as trotterised pauli expoentials
            # (exp (-ha Pa) * exp (-hb Pb) = exp (-ha Pa - hb Pb))
            self._trotter_step = expm(-1j * self._qubit_operator * self._time_step)

        self._evolved_measurements = {}

    def _measure(self, trotter_evolution):
        return [numpy.vdot(trotter_evolution, operator.dot(trotter_evolution)).real.item() for operator in self._measurements]

    def _trotter_stepper(self):
        for t in self._time_space:
            if t == 0:
                trotter_evolution = self._initial_state
            else:
                trotter_evolution = self._trotter_step * trotter_evolution
                if (self._evolution_type == TimeEvolutionType.IMAG):  # Renorm at each step
                    trotter_evolution = trotter_evolution / numpy.linalg.norm(trotter_evolution)
            self._evolved_measurements[t] = self._measure(trotter_evolution)

    def execute(self):
        self._trotter_stepper()

