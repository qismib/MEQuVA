# example.py for latest qiskit version.
#
# qiskit==0.28.0
# qiskit-aer==0.8.2
# qiskit-ibmq-provider==0.15.0
# qiskit-nature=0.1.5
# qiskit-terra=0.18.0


#import matplotlib.pyplot as plt
import numpy as np

from qiskit_nature.drivers import PySCFDriver, UnitsType, HFMethodType, Molecule
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization import QubitConverter
#from qiskit_nature.circuit.library.ansatzes import UCCSD
from qiskit_nature.circuit.library import HartreeFock

from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver

from qiskit.algorithms.optimizers import L_BFGS_B, CG
from qiskit.algorithms import VQE

from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal

from qiskit.utils import QuantumInstance

from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit


def add_unitary_gate(circuit,
                     qubit1,
                     qubit2,
                     params,
                     p0):
    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2,qubit1)
    circuit.u(params[p0],params[p0+1],params[p0+2],qubit1); p0 += 3
    circuit.u(params[p0],params[p0+1],params[p0+2],qubit2); p0 += 3
    circuit.cx(qubit2,qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2)

#definition of SO(4) variational form for arbitrary nuber of qubits

def SO_04(numqubits, init = None):
        num_parameters = 6*(numqubits-1)
        parameters = []
        for i in range(num_parameters):
            name = "par" + str(i)
            new = Parameter(name)
            parameters.append(new)
        q = QuantumRegister(numqubits, name='q')
        if init != None:
            circ = init
        else:
            circ = QuantumCircuit(q)

        if numqubits == 4:
            add_unitary_gate(circ, 0, 1, parameters, 0)
            add_unitary_gate(circ, 2, 3, parameters, 6)
            add_unitary_gate(circ, 1, 2, parameters, 12)

        return circ


def vqe_function(geometry,
                 spin = 0,
                 charge = 1,
                 basis = 'sto6g',
                 var_form_type = 'TwoLocal',
                 quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator')),
                 optimizer = CG(),
                 converter = QubitConverter(mapper = ParityMapper(),
                                            two_qubit_reduction = True)
                 ):
    #setting up molecule driver

    driver = PySCFDriver(atom = geometry,
                         unit = UnitsType.ANGSTROM,
                         basis = basis,
                         spin = spin,
                         charge = charge,
                         hf_method = HFMethodType.RHF)

    problem = ElectronicStructureProblem(driver)
    main_op = problem.second_q_ops()[0]

    #EXPLICIT DATA IS REQUIRED
    alpha, beta = 1, 1 #doing some bonanza to resolve conflicts
    num_particles = (alpha, beta)
    num_spin_orbitals =3*2

    qubit_op = converter.convert(main_op, num_particles=num_particles)

    init_state = HartreeFock(num_spin_orbitals,
                             num_particles,
                             converter)

    init_state.draw()

    #ansatz

    if var_form_type == 'TwoLocal':
        ansatz = TwoLocal(num_qubits = num_spin_orbitals - 2,
                          rotation_blocks = 'ry',
                          entanglement_blocks = 'cx',
                          initial_state = init_state,
                          entanglement = 'linear')
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         quantum_instance = quantum_instance)
    elif var_form_type == 'UCCSD':
        vqe_solver = VQEUCCFactory(quantum_instance,
                                   initial_state = init_state)

    elif var_form_type == 'SO(4)':
        ansatz = SO_04(num_spin_orbitals - 2,
                       init = init_state)
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         quantum_instance = quantum_instance)

    #set up VQE

    calc = GroundStateEigensolver(converter, vqe_solver)
    elec = calc.solve(problem)
#    print(res)


    print("RISULTATO: ",
          "\n\tEnergia calcolata: ", elec.computed_energies[0],
          "\n\tEnergia ioni: ", elec.nuclear_repulsion_energy,
          "\n\tTotale: ", elec.total_energies[0],
          "\n\tNum particelle: ", elec.num_particles,
          "\n\tSpin: ", elec.spin)

#main

dist = str(0.87)
dist2 = str (0.435)
alt = str(np.sqrt(0.87**2 - 0.435**2))
geometry = 'H 0. 0. 0.; H 0. 0. ' + dist + '; H 0. ' + alt + " " + dist2

vqe_function(geometry, var_form_type='TwoLocal')
vqe_function(geometry, var_form_type='SO(4)')
vqe_function(geometry, var_form_type='UCCSD')
