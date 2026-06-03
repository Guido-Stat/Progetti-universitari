# Progetti-universitari
Repository in cui sono presenti i progetti universitari che ho portato a termine.

Al suo interno sono presenti 3 cartelle che rappresentano 3 differenti insegnamenti universitari, organizzate come segue: 

## Data Mining

| Progetto | Descrizione e Tecniche Utilizzate |
| :--- | :--- |
| `Homework1` | Sviluppo di una **rete neurale** per la classificazione di una variabile target binaria. Il dataset di partenza risulta fortemente sbilanciato; sono state pertanto implementate e documentate specifiche tecniche di campionamento e bilanciamento per ottimizzare le metriche di valutazione del modello. |
| `Homework2` | Progettazione e addestramento di una **Rete Neurale Convoluzionale (CNN)** dedicata alla classificazione di immagini. All'architettura personalizzata è stato affiancato l'utilizzo di una rete pre-addestrata (Transfer Learning), sulla quale è stata eseguita una procedura di **Fine-Tuning** per migliorarne l'accuratezza sul dominio specifico. |
| `Homework_finale` | Sviluppo end-to-end di una **rete neurale concatenata** (Multi-Input/Complex Architecture) costruita interamente da zero per la classificazione avanzata di immagini. |

<br>

## Gestione ed Elaborazione di Big Data

| Progetto | Pipeline di Sviluppo ed Architettura |
| :--- | :--- |
| `recommender_system` | Implementazione di un sistema di raccomandazione di film basato su un dataset di utenti, pellicole e valutazioni. Il cuore del progetto prevede l'applicazione dell'algoritmo **ALS (Alternating Least Squares)** distribuito tramite **Apache Spark** e l'integrazione con il database NoSQL **MongoDB**. |

### Dettaglio delle Fasi del Progetto `recommender_system`

L'architettura del sistema è suddivisa in quattro moduli sequenziali e interconnessi:

1. **Ingegnerizzazione e Caricamento Dati (ETL)** I dati grezzi vengono estratti, trasformati e strutturati tramite **PySpark**. In questa fase il dataset viene arricchito generando nuove feature dalle collezioni esistenti o integrando informazioni dal set originale, per poi procedere al caricamento massivo su **MongoDB**.

2. **Addestramento del Modello** Configurazione e calcolo dell'algoritmo di fattorizzazione matriciale **ALS** sfruttando la potenza del calcolo distribuito di **PySpark** per gestire l'interazione sparsa utenti-item.

3. **Generazione e Stoccaggio delle Raccomandazioni** Produzione delle raccomandazioni personalizzate per l'utenza sulla base dei fattori latenti appresi dal modello. Gli output predittivi vengono formattati e salvati nuovamente su **MongoDB** per garantirne la persistenza.

4. **Interrogazione e Analisi dei Risultati** Implementazione di una sezione di test dedicata alle query sul database NoSQL, volta a simulare casi d'uso reali e a dimostrare l'efficacia e la rapidità di recupero delle raccomandazioni generate.
