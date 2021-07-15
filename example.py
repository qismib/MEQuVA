'''
EXAMPLE: VQE IN ACTION

docs: https://github.com/qismib/MEQuVA

'''

#import things, some not used

import matplotlib.pyplot as plt
import numpy as np
import pprint

from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SLSQP, COBYLA, CG, L_BFGS_B
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.applications import MolecularGroundStateEnergy
from qiskit.chemistry.algorithms import ground_state_solvers
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import QubitMappingType, Hamiltonian, TransformationType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.operators import PauliExpectation
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import IBMQ

provider = IBMQ.load_account()   #saved account required

#----------------------------------------------------------------------------------------

#definition of SO(4) operator

def add_unitary_gate(circuit,qubit1,qubit2,params,p0):
    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2,qubit1)
    circuit.u3(params[p0],params[p0+1],params[p0+2],qubit1); p0 += 3
    circuit.u3(params[p0],params[p0+1],params[p0+2],qubit2); p0 += 3
    circuit.cx(qubit2,qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2) 

#definition of SO(4) variational form for arbitrary nuber of qubits

class SO_04(VariationalForm):
    
    def __init__(self, numqubits):
        self._num_qubits = numqubits
        self._num_parameters = 6*(numqubits-1)
               
    def construct_circuit(self, parameters):
        q = QuantumRegister(self._num_qubits, name='q')
        circ = QuantumCircuit(q)
                
        if self._num_qubits == 4:        
            add_unitary_gate(circ, 0, 1, parameters, 0)
            add_unitary_gate(circ, 2, 3, parameters, 6)
            add_unitary_gate(circ, 1, 2, parameters, 12)
            
        return circ
    
    @property
    def num_parameters(self):
        return self._num_parameters    

#----------------------------------------------------------------------------------------

#function to initialize and run vqe

def vqe_function(geometry, basis = 'sto-6g', var_form_type = 'TwoLocal(ry)', 
                 quantum_instance = Aer.get_backend("statevector_simulator"),
                 optimizer = CG(maxiter=1000),
                 callback_tf=True):
        
    two_qubit_reduction = True
    qubit_mapping = 'parity'       
		#see https://qiskit.org/documentation/stubs/qiskit.chemistry.core.QubitMappingType.html?highlight=qubitmappingtype#qiskit.chemistry.core.QubitMappingType
    
    charge = 1     #data for H3+
    spin = 0


    driver = PySCFDriver(atom = geometry,
                         unit=UnitsType.ANGSTROM, spin = spin,
                         charge=charge, basis=basis,hf_method=HFMethodType.RHF)    #using RHF because of H3+
    
    molecule = driver.run()
    
    shift = molecule.nuclear_repulsion_energy                #vqe finds just electronic energy
    num_particles = molecule.num_alpha + molecule.num_beta   #alpha/beta are electrons with spin up and down
    num_orbitals = molecule.num_orbitals
    num_spin_orbitals = num_orbitals*2
    
    
    core = Hamiltonian(transformation=TransformationType.FULL, qubit_mapping=QubitMappingType.PARITY,    #change qubit mapping carefully
                   two_qubit_reduction=two_qubit_reduction, freeze_core=False)

    qubitOp, aux_ops = core.run(molecule)
    
    initial_state = HartreeFock(num_spin_orbitals, 
                                num_particles, 
                                qubit_mapping,
                                two_qubit_reduction)
    
    var_form = TwoLocal() #dummy
    
    if var_form_type == 'TwoLocal(ry)':
        var_form = TwoLocal(rotation_blocks=['ry'], entanglement_blocks='cx',
                            num_qubits=4,
                            reps=2,
                            initial_state=initial_state,
                            entanglement='linear')
        
    if var_form_type == 'TwoLocal(ry_rz)':
        var_form = TwoLocal(rotation_blocks=['ry','rz'], entanglement_blocks='cz',
                            num_qubits=4,
                            initial_state=initial_state,
                            entanglement='linear')
    
    if var_form_type == 'UCCSD':
        var_form = UCCSD(num_orbitals=num_spin_orbitals,
                        num_particles=num_particles,
                        initial_state=initial_state,
                        qubit_mapping=qubit_mapping,
                        two_qubit_reduction=two_qubit_reduction,
                        z2_symmetries=None)
        
    if var_form_type == 'SO(4)':
        var_form = SO_04(4)
    
    init_parm = np.random.rand(var_form.num_parameters)        #starting with random paraeters for the variational form


    #callback function definition to store intermediate result

    counts = []
    values = []
    std_var = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        std_var.append(std)
    
    if callback_tf == True:
        vqe = VQE(qubitOp, var_form, optimizer, initial_point=init_parm, 
                  callback=store_intermediate_result, expectation=PauliExpectation())
        vqe_result_tot = vqe.run(quantum_instance)
    
        result = np.real(vqe_result_tot['eigenvalue'])
        err = 0
    
        for i in range(len(std_var)):
            if values[i] == result:
                err = std_var[i]
                break
        
        return np.real(vqe_result_tot['eigenvalue'] + shift) , err, vqe_result_tot,vqe
    
    else:
        vqe = VQE(qubitOp, var_form, optimizer, initial_point=init_parm)
        vqe_result_tot = vqe.run(quantum_instance)
        return np.real(vqe_result_tot['eigenvalue'] + shift)

#----------------------------------------------------------------------------------------


#importing noise model from ibmq_santiago

backend = Aer.get_backend("qasm_simulator")
device = provider.get_backend("ibmq_santiago")

coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates
seed = 150
aqua_globals.random_seed = seed

quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                    coupling_map=coupling_map, noise_model=noise_model,shots=8000)


#H3+ geometry def

dist = 1.0
alt=np.sqrt(dist**2 - (dist/2)**2)
geometry = "H .0 .0 .0; H .0 .0 " + str(dist) + "; H .0 " + str(alt) + " " + str(dist/2) 

result = vqe_function(geometry, var_form_type="UCCSD", 
                        quantum_instance=quantum_instances,
                        callback_tf=False)

print("VQE result = ", result)