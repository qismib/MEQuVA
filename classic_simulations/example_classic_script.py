import numpy as np
from pyscf import gto,scf,ao2mo,mp,cc,fci,tools
import matplotlib.pyplot as plt

'''
Iniziamo creando la molecola a cui siamo interessati.
In particolare ci serve:
- la posizione degli atomi
- la carica complessiva della molecola
- lo spin totale degli elettroni
- la base che vogliamo usare per fare questi conti
'''

### Definiamo un vettore di distanze a cui vogliamo posizionare i nostri atomi di idrogeno
dist = np.arange(0.3, 4, .05)
alt=np.sqrt(dist**2 - (dist/2)**2)

# E una base per il nostri calcoli
basis = '6-31g' #'sto-6g' '6-31g' 'cc-pvdz' 'aug-cc-pvdz' ,...



# Creiamo dei contenitori per le energie misurate ad ogni geometria
energies = {}

energies["HF"] = []
energies["FCI"] = []
energies["MP2"] = []
energies["CCSD"] = []

for i in range(len(dist)):

	#geometry = "H .0 .0 .0; H .0 .0 " + str(dist[i]) + "; H .0 " + str(alt[i]) + " " + str(dist[i]/2)
	geometry = "H .0 .0 .0; H .0 .0 .742; H .0 " + str(dist[i]) + " " + str(0.742/2)
	mol = gto.M(atom=geometry,charge=1,spin=0,basis=basis,symmetry=True,verbose=0)


	# A questo punto si fa un conto di campo medio (Hartree-Fock)
	# Se gli elettroni con spin up spno tanti quanti gli elettroni
	# con spin down (spin totale 0) si fa un Restricted Hartree Fock

	# Per maggiori informazioni su RHF,ROHF, UHF vedere 

	# Szabo, Ostlund - Modern Quantum Chemistry <- ottimo!
	# https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method
	# https://en.wikipedia.org/wiki/Restricted_open-shell_Hartree%E2%80%93Fock
	# https://en.wikipedia.org/wiki/Unrestricted_Hartree%E2%80%93Fock

	mf  = scf.RHF(mol)
	Ehf = mf.kernel() #<- chiamando il kernel otteniamo l'energia calcolata dal metodo
	energies["HF"].append(Ehf)


	# Il conto di campo medio è la prima approsimazione che si può fare nello studio
	# della nostra molecola. Da questo possiamo partire per studiare le correlazioni 
	# tra gli elettroni a diversi livelli


	## FCI

	# Quando il sistema ha pochi elettroni, possiamo permetterci di diagonalizzare esattamente 
	# l' Hamiltoniano del sistema nella base degli orbitali scelta.
	# Questo metodo prende il nome di Full Configuration Interaction (FCI)

	#### NB queste righe vanno commentate se la base diventa troppo grande

	fci_h3 = fci.FCI(mf)  #<- nei metodi correlati passiamo come argomento un conto di campo medio, HF
	e_fci = fci_h3.kernel()[0]
	energies["FCI"].append(e_fci)

	# Sfortunatamente, questo metodo diventa particolarmente oneroso quando incrementiamo 
	# la taglia della base o il numero di elettroni, dobbiamo queindi ricorrere a delle 
	# approssimazioni

	## MP2

	# Possiamo ad esempio aggiungere perturbativamente le interazioni tra gli elettroni. Questa tecnica 
	# prende il nome di Møller-Plesset. Nel nostro caso consideriamo una perturbazione al secondo ordine

	mp2   = mp.MP2(mf)
	e_mp2 = mp2.kernel()[0]
	e_mp2 += Ehf # <- questa linea è necessaria perchè MP2 generalmente mostra l'energia di differenza con HF
	energies["MP2"].append(e_mp2)

	#### CCSD
	# Un' approssimazione famosa è quella di Coupled Cluster (https://en.wikipedia.org/wiki/Coupled_cluster)
	# in cui facciamo un'ipotesi (ansatz) sulla forma della funzione d'onda del sistema, ipotizzando che siano
	# presenti solo alcuni tipi di eccitazioni elettroniche

	# Quando ci restringiamo ad eccitazioni elettroniche singole e doppie, questo metodo prende il nome di
	# Coupled Cluster with Singles and Doubles (CCSD). Nel nostro caso, con 2 elettroni abbiamo che tutte le 
	# eccitazioni possibili sono solo singole o doppie, quindi CCSD == FCI

	# Il metodo CCSD(T), nel quale le eccitazioni triple vengono poi aggiunte perturbativamente, è considerato 
	# il "golden standard" della chimica computazionale (ha fornito alcune delle simulazioni più precise)

	ccsd_h3 = cc.CCSD(mf)
	e_ccsd  = ccsd_h3.kernel()[0]
	e_ccsd += Ehf # <- questa linea è necessaria perchè CCSD generalmente mostra l'energia di differenza con HF
	energies["CCSD"].append(e_ccsd)


	# Esistono molti altri metodi (simulazioni in piccoli sottospazi attivi, ad esempio) Ma per il momento questi 
	# possono darci una panoramica della superficie energetica della molecola




## Infine, plottiamo i risultati

for (method,E) in energies.items():
	plt.plot(dist,E,label=method)


#plt.plot(alt,energies["HF"])
plt.xlabel(r"$d$ $[\AA]$")
plt.ylabel(r"Energy $[Ha]$")
plt.legend()
plt.show()
