## In questo script ptoveremo ad usare il tapering per trovare le simmetrie
## dell'operatore Hamiltoniano relativo al sistema H3+ e ridurrre il numero 
# di qubits richiesti

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt


from qiskit import QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import NumPyEigensolver, VQE
from qiskit.aqua.components.optimizers import SLSQP, COBYLA, CG
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.applications import MolecularGroundStateEnergy
from qiskit.chemistry.algorithms import ground_state_solvers
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType


import qiskit
from qiskit                            import IBMQ    
from qiskit                            import BasicAer,Aer    
from qiskit.aqua                       import set_qiskit_aqua_logging, QuantumInstance
from qiskit.aqua.operators             import Z2Symmetries, WeightedPauliOperator
from qiskit.aqua.components.optimizers import L_BFGS_B,CG,SPSA,SLSQP, COBYLA


from qiskit.chemistry.components.initial_states                    import HartreeFock

from qiskit.chemistry.core                                         import TransformationType, QubitMappingType, Hamiltonian
from qiskit.chemistry                                              import set_qiskit_chemistry_logging,qmolecule
from qiskit.chemistry.components.variational_forms                 import UCCSD

if __name__ == "__main__":
	dist = np.arange(0.3, 4, .1)
	alt=np.sqrt(dist**2 - (dist/2)**2)

	j = 5
	geometry = "H .0 .0 .0; H .0 .0 " + str(dist[j]) + "; H .0 " + str(alt[j]) + " " + str(dist[j]/2)



	driver            = PySCFDriver(atom=geometry, unit=UnitsType.ANGSTROM,charge=1,spin=0,basis='sto-6g',hf_method=HFMethodType.RHF)
	molecule          = driver.run()

	
	core              = Hamiltonian(qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,freeze_core=False) 
	qubit_op, aux_ops = core.run(molecule)
	aux_ops           = aux_ops[:3]
	aux_ops.append(qubit_op)
	dE      = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy

	print("ENERGY SHIFTS")
	print(core._energy_shift,core._ph_energy_shift,core._nuclear_repulsion_energy)
	print("TOTAL SHIFT")
	print(core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy)
	

	### Altrimenti, senza Hamiltonian
	'''
	ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)    
	qubit_op = ferOp.mapping(map_type='parity')
	'''

	ee  = NumPyEigensolver(qubit_op,k=1)
	print("FCI in active space energy ",ee.run()['energy'] + (dE))


	 # Now search for symmetries

	print("====================================================")
	print("Initial number of qubits: "+str(qubit_op.num_qubits))
	print("====================================================")
	    
	print("Finding symmetries...")

	z2_symmetries   = Z2Symmetries.find_Z2_symmetries(qubit_op)

	nsym            = len(z2_symmetries.sq_paulis)
	the_tapered_op  = qubit_op
	the_tapered_aux = aux_ops
	sqlist          = None
	z2syms          = None

	    
	if(nsym>0):
		print(z2_symmetries)
		tapered_ops        = z2_symmetries.taper(qubit_op)
		smallest_eig_value = 99999999999999
		smallest_idx       = -1

		for idx in range(len(tapered_ops)):
			ee  = NumPyEigensolver(tapered_ops[idx], k=1)
			curr_value = ee.run()['energy']
			print(">>> tapering, energy ",curr_value+dE)
			if curr_value < smallest_eig_value:
				smallest_eig_value = curr_value
				smallest_idx = idx

		the_tapered_op = tapered_ops[smallest_idx]
		sqlist = the_tapered_op.z2_symmetries.sq_list
		z2syms = the_tapered_op.z2_symmetries

		# Bisogna fare il tapering anche agli operatori ausiliari se vogliamo misurarli
		
		the_tapered_aux = []
		for a in aux_ops:
			if(not a.is_empty()):
				the_tapered_aux.append(the_tapered_op.z2_symmetries.taper(a))
			else:
				the_tapered_aux.append(a)
		

	elif(nsym==0):
		print("No symmetries found")


	print("====================================================")
	print("Final number of qubits: "+str(the_tapered_op.num_qubits))
	print("====================================================")

