import numpy as np
from pyscf import gto,scf,ao2mo,mp,cc,fci,tools
import matplotlib.pyplot as plt

'''
Studiamo la forma della superficie energetica al variare della base di orbitali scelta
'''

### Definiamo un vettore di distanze a cui vogliamo posizionare i nostri atomi di idrogeno
dist = np.arange(0.3, 4, .1)
alt=np.sqrt(dist**2 - (dist/2)**2)

# E una base per il nostri calcoli
basis = ['sto-6g','6-31g','cc-pvdz','aug-cc-pvdz'] 



# Creiamo dei contenitori per le energie misurate ad ogni geometria
energies = {}


for (j,base) in enumerate(basis):

	energies[base] = []

	for i in range(len(dist)):

		geometry = "H .0 .0 .0; H .0 .0 " + str(dist[i]) + "; H .0 " + str(alt[i]) + " " + str(dist[i]/2)
		mol = gto.M(atom=geometry,charge=1,spin=0,basis=base,symmetry=True,verbose=0)

		mf  = scf.RHF(mol)
		Ehf = mf.kernel() #<- chiamando il kernel otteniamo l'energia calcolata dal metodo


		ccsd_h3 = cc.CCSD(mf)
		e_ccsd  = ccsd_h3.kernel()[0]
		e_ccsd += Ehf # <- questa linea è necessaria perchè CCSD generalmente mostra l'energia di differenza con HF
		energies[base].append(e_ccsd)
	print(energies[base])




## Infine, plottiamo i risultati

for (base,E) in energies.items():
	plt.plot(dist,E,label=base)


#plt.plot(alt,energies["HF"])
plt.xlabel(r"$d$ $[\AA]$")
plt.ylabel(r"Energy $[Ha]$")
plt.legend()
plt.show()