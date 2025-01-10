# Documentazione Classe RiskMonitoring

## Descrizione Generale
La classe `RiskMonitoring` implementa un sistema di monitoraggio del rischio per un robot che esegue operazioni di navigazione, raccolta (picking) e lancio (throwing) di oggetti. Il sistema valuta le variazioni di rischio tra stati precedenti e aggiornati per determinare se è necessario ripianificare le operazioni.

## Funzionalità Principali

### 1. Gestione degli Scenari
- Organizza e ordina gli scenari operativi in base alla loro probabilità
- Filtra le azioni in tre categorie principali:
  - Azioni di navigazione (incluse attese e rilasci)
  - Azioni di lancio (throwing)
  - Azioni di raccolta (picking)

### 2. Analisi del Rischio
- Calcola le deviazioni di rischio confrontando:
  - Mappe di rischio precedenti vs nuove
  - Tempi di esecuzione
  - Probabilità di successo delle operazioni
- Utilizza un sistema di pesi per bilanciare:
  - Tempo (20% del peso totale)
  - Rischio (80% del peso totale)

### 3. Valutazione per Rischedulazione
- Monitora le deviazioni di rischio per ogni tipo di operazione
- Confronta le deviazioni con una soglia massima (20% di default)
- Determina se è necessario ripianificare le operazioni
- Fornisce metriche di rischio adattate per l'interfaccia utente

## Parametri Chiave
- `t_weight`: 0.2 (peso del fattore tempo)
- `r_weight`: 0.8 (peso del fattore rischio)
- `max_deviation`: 20% (deviazione massima consentita prima della ripianificazione)

## Processo di Valutazione del Rischio
1. Acquisizione degli scenari e delle loro probabilità
2. Calcolo delle deviazioni di rischio per ogni tipo di azione
3. Ponderazione delle deviazioni in base alle probabilità degli scenari
4. Normalizzazione dei valori di rischio per la visualizzazione UI
5. Determinazione della necessità di ripianificazione

## Output
Il sistema produce diversi valori di rischio:
- Rischi di navigazione (vecchi e nuovi)
- Rischi di manipolazione (vecchi e nuovi)
- Indicatore booleano per la necessità di ripianificazione
