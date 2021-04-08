# VQE Qiskit implementation

Questa repository contiene...

## Descrizione dei file caricati

* ***VQE Implementation in Qiskit.ipynb***: notebook preso dai tutorial di qiskit e solo leggermente modificato. Applicazione di VQE per LiH.

* ***VQE H3.ipynb***: è il primissimo passo fatto da me, codice pieno di annotazioni e parti commentatate via. Quando viene fatto partire ricalcola molteplici volte le stesse cose ed è quindi molto lento.

* ***H3+ energy - rounded.ipynb***: sostanzialmente è l'ottimizzazione del notebook precedente: può avere ancora qualche problema, ma funziona.

* ***NoteTesi.pdf***: bozza in latex che spiega alcuni degli elementi incontrati nel codice. Mi è necessaria per capire bene cosa sono alcune cose (ad esempio UCCSD).

* ***requirements.txt***: lista dei pacchetti installati dell'ambiente virtuale (versione di Python: 3.8.5).

## Problemi / questioni in sospeso

* Sia che io imposti uno spin S (di H3+) uguale a 0 o uguale a 1 Qiskit me lo mette a 0, perchè?

* L'energia dello stato fondamentale, così come la lunghezza di legame, non è precisamente uguale a quella tabulata (o almeno così pare).

* Da implementare una geometria di H3+ lineare e non triangolare.

* Da aggiungere commenti esplicativi nei notebook.

* Da rimuovere i notebook inutili. Devo prima capire bene cosa mi serve, ma poi fanno solo confusione.

## Stato dell'arte
