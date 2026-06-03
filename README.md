# Progetti-universitari
Repository in cui sono presenti i progetti universitari che ho portato a termine.

Al suo interno sono presenti 3 cartelle che rappresentano 3 differenti insegnamenti universitari, organizzate come segue: 

| Data mining |
| ---------------------- |
| Homework1: progetto in cui viene realizzata una rete neurale per la classificazione di una variabile target binaria. Il dataset è sbilanciato, quindi vengono usate delle tecniche per far fronte a questa problematica     
| Homework2: progetto in cui viene costruita una rete neurale convoluzionale per la classificazione di immagini. Viene anche usata una rete neurale pre addestrata sulla quale viene eseguito del fine tuning. 
| Homework_finale: progetto in cui si costruisce da zero una rete neurale concatenata per la classificazione di immagini.

| Gestione ed elaborazione di Big Data |
| ---------------------- |
| recommender_system: progetto in cui viene implementato un algoritmo che permette di ottenere delle
raccomandazioni da un data set contenente gli utenti, i film e le votazioni
Per farlo viene implementato l'algoritmo ALS (Alternate Least
Squares) tramite Spark. Il codice è suddiviso in tre fasi fondamentali più un'ultima parte in cui si va
a interrogare il database.
- La prima fase consiste nell'andare a caricare su MongoDB il set di dati; per fare
ciò i dati subiranno prima delle trasformazioni tramite PySpark e il database verrà
inoltre arricchito con ulteriori dati (recuperati dal set originale o creati a partire
dalle altre collezioni).
- La seconda fase prevede invece l'implementazione tramite PySpark dell'algoritmo
prima citato.
- Infine la terza fase prevede la produzione di raccomandazioni sulla base dei
risultati ottenuti grazie all'algoritmo e al loro caricamento, sempre su MongoDB.
Infine è prevista una piccolissima parte in cui si interroga il database, così
da mostrare dei potenziali usi per quanto prodotto.
