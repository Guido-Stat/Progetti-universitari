"""
Il seguente codice vuole implementare un algoritmo che permetta di ottenere delle
raccomandazioni da un data set contenente gli utenti, i film e le votazioni
Per farlo viene implementato l'algoritmo ALS (Alternate Least
Squares) tramite Spark. Il data set utilizzato è quello citato nell'articolo
'Recommendation System for E-commerce Using Alternating Least Squares (ALS) on
Apache Spark', ossia il Movielens 10M, il quale contiene 71.567 utenti (anche se poi
effettivi sono 89878) e 10.681 film, per un totale di 10.000.054 valutazioni.
Il codice è suddiviso in tre fasi fondamentali più un'ultima parte in cui si va
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
"""

# Come prima cosa importiamo tutti i moduli necessari per il codice
import pymongo as mongo
import pyspark as ps
import os
import json
import datetime
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.evaluation import RegressionMetrics

"""
PRIMA FASE: caricamento data set su MongoDB
"""
# Per vedere in quanto tempo la prima fase verrà completata, prendiamo l'ora
# corrente, la quale verrà poi sottratta all'ora della fine dell'esperimento
# per ottenere il tempo impiegato.
ora_inizio_P1 = datetime.datetime.now()
print(f"Prima fase iniziata alle ore {ora_inizio_P1}")

# Come prima cosa prendiamo il percorso della cartella con cui stiamo lavorando.
# In essa sarà presente lo script, il data set e tutti i file che verranno prodotti
# dal codice. Questo viene fatto con una funzione elementare del pacchetto os.
percorso = os.getcwd()

# Istanziamo il framework Spark in locale, così che esso utilizzerà i processori
# del nostro computer. Indicando il carattere * tra parentesi quadre,
# gli diciamo di usarli tutti.
sc = ps.SparkContext("local[*]", appName="Recommendation system using ALS")
print("Framework PySpark istanziato.")

# Carichiamo come RDD il contenuto del file ove è presente il nostro data set.
# In questo caso ogni riga del file verrà rappresentata come una stringa
# e ogni partizione sarà elaborata in parallelo da Spark.
print("Caricamento file ratings.dat...")
dRating = sc.textFile(percorso + "/ml-10M100K/ratings.dat")

# Vediamo quindi come è strutturato il file
print(f"File caricato. Esso è strutturato come segue: {dRating.take(5)}")

# A questo punto notiamo che all'interno del file sono presenti 4 colonne:
# 1: UserID, 2: MovieId, 3: Rating, 4: TimeStamp.
# Ogni riga è rappresentata come una stringa, quindi per poterle separare
# usiamo una trasformazione map. Nel nostro caso, poiché nel file le colonne
# sono separate da :: (una coppia di due punti), usiamo il metodo split.
print("Trasformazioni ratings in corso...")
dRating1 = dRating.map(lambda x: (x.split("::")))

# Oltre a quanto sopra, possiamo notare che ai nostri fini la quarta colonna
# non è rilevante, motivo per il quale la rimuoviamo. Lo facciamo sempre
# tramite una trasformazione map, la quale prende l'RDD precedente in
# input, e restituisce un nuovo RDD in cui sono presenti solo le prime
# 3 colonne.
dRating2 = dRating1.map(lambda x: (x[0], x[1], x[2]))

# A questo punto, poiché vogliamo caricare il nostro set di dati sul
# database NoSQL MongoDB, trasformiamo ogni riga in un dizionario.
# Ricordiamo infatti che MongoDB lavora tramite documenti espressi in
# formato Json.
# Inoltre, poiché l'algoritmo usa la funzione ALS, essa lavora con dati
# interi (colonna degli identificativi di utenti e film) e decimali
# (colonna delle valutazioni), motivo per il quale effettuiamo
# subito le trasformazioni, così che poi possiamo salvare i dati in formato
# corretto su MongoDB.
dRating3 = dRating2.map(lambda x: {
    "UserId": int(x[0]),
    "MovieId": int(x[1]),
    "Rating": float(x[2])})
print("Trasformazioni ratings completate.")

# Possiamo ora procedere anche con il caricamento del file movies.dat.
# Differentemente dal precedente, questo non ci servirà per l'algoritmo,
# tuttavia, lo implementiamo sul database, così da avere un quadro più completo
# e ricco di informazioni.
print("Caricamento file movies.dat...")
dMovies = sc.textFile(percorso + "/ml-10M100K/movies.dat")

# Come prima, osserviamo come è strutturato il file:
print(f"File caricato. Esso è strutturato come segue: {dMovies.take(5)}")

# A questo punto notiamo che all'interno del file sono presenti 3 colonne:
# 1: MovieID, 2: Title, 3: Genres.
# Come per il primo file, anche in questo attueremo delle trasformazioni. Come
# prima cosa separiamo le colonne.
print("Trasformazioni movies in corso...")
dMovies1 = dMovies.map(lambda x: x.split("::"))

# Andiamo poi a trasformare ogni riga in un dizionario. L'ultimo elemento di ogni
# riga rappresenta i generi. Poiché ogni film ha più generi, li separiamo con il
# comando split. In questo modo quando salveremo la collezione su MongoDB avremo
# che i generi saranno rappresentati come una lista di valori.
dMovies2 = dMovies1.map(lambda x: {
    "MovieId": int(x[0]),
    "Title": str(x[1]),
    "Genres": str(x[2].split("|"))
})
print("Trasformazioni movies completate.")

# Come ulteriore passaggio per arricchire il nostro database, possiamo creare
# una collection Users, la quale conterrà gli utenti e i vari film che essi hanno
# visto. I film saranno rappresentati come una collezione di valori. Questo permette di
# associare immediatamente a ogni utente i film che esso ha visto, senza
# dover necessariamente passare dalla collection Ratings, la quale ha oltre
# 10 milioni di documenti.
# Come prima cosa, prendiamo il dizionario dei rating e lo trasformiamo in una
# tupla, contente lo UserId e il MovieId. Il rating non ci interessa, quindi non
# lo consideriamo.
print("Trasformazioni users in corso...")
dUsers = dRating3.map(lambda x: (x["UserId"], x["MovieId"]))

# A questo punto possiamo raggruppare tutti gli utenti con lo stesso ID, e
# unire in un'unica collezione tutti i film che l'utente ha visionato.
# Facciamo questo con groupByKey, il quale è un comando simile a reduceByKey,
# ma che viene impiegato quando non ci sono aggregazioni da dover fare.
# Con mapValues indichiamo che i valori saranno contenuti in una lista.
dUsers2 = dUsers.groupByKey().mapValues(list)

# Poiché l'operazione groupByKey ha mescolato l'ordine delle righe, le andiamo
# a ordinare in ordine crescente prendendo come riferimento l'Id dell'utente.
dUsers2 = dUsers2.sortBy(lambda x: x[0])

# Per aggiungere informazione al database, possiamo contare quanti film ha visto
# ciascun utente. Di fatto, il numero di film visti coincide con la lunghezza della
# lista di ciascun utente contenente gli Id dei film visti dallo stesso, quindi
# ottenere tale informazione non è difficile.
dUsers3 = dUsers2.map(lambda x: (x[0], len(x[1]), x[1]))

# Infine, come per le altre raccolte, trasformiamo il tutto in un dizionario.
dUsers4 = dUsers3.map(lambda x: {
    "UserId": x[0],
    "Total Movies seen": x[1],
    "MovieId": x[2]
})
print("Trasformazioni users completate.")

# Prima di procedere con il caricamento su MongoDB, osserviamo che esso si
# aspetta dei documenti in formato Json, tuttavia nel nostro caso, nel momento
# in cui andremo a salvare i file, essi verranno salvati come file di testo.
# Procediamo quindi alla conversione usando una semplice funzione della libreria json,
# così che essi vengano salvati con il formato corretto.
dRating4 = dRating3.map(lambda x: json.dumps(x))
dMovies3 = dMovies2.map(lambda x: json.dumps(x))
dUsers5 = dUsers4.map(lambda x: json.dumps(x))

# A questo punto possiamo salvare le RDD sul nostro computer. In base alla
# dimensione Spark dividerà il set di dati in più file.
dRating4.saveAsTextFile(percorso + "/Ratings_json")
print("File ratings salvati correttamente.")
dMovies3.saveAsTextFile(percorso + "/Movies_json")
print("File movies salvati correttamente.")
dUsers5.saveAsTextFile(percorso + "/Users_json")
print("File users salvati correttamente.")

# A questo punto possiamo stabilire la connessione con MongoDB:
client = mongo.MongoClient("mongodb://localhost:27017/")
print("Connessione con MongoDB stabilita.")

# Osserviamo la lista di file di testo salvati da Spark.
lista_file_ratings = os.listdir(percorso + "/Ratings_json")
print(f"L'elenco dei file salvati per i ratings è il seguente: {lista_file_ratings}")
lista_file_movies = os.listdir(percorso + "/Movies_json")
print(f"L'elenco dei file salvati per i movies è il seguente: {lista_file_movies}")
lista_file_users = os.listdir(percorso + "/Users_json")
print(f"L'elenco dei file salvati per gli users è il seguente: {lista_file_users}")

# Prima di procedere con il caricamento notiamo che nella cartella in cui sono finiti
# i file di testo salvati da Spark, sono stati generati anche dei file che a noi non
# interessano. Pertanto, li eliminiamo.
print("Eliminazione file inutili...")
lista_file_ratings_finale = []
for elemento in lista_file_ratings:
    if (".part" not in elemento and "._SUCCESS" not in elemento and "_SUCCESS"
            not in elemento):
        lista_file_ratings_finale.append(elemento)
lista_file_ratings_finale = sorted(lista_file_ratings_finale)

lista_file_movies_finale = []
for elemento in lista_file_movies:
    if (".part" not in elemento and "._SUCCESS" not in elemento and "_SUCCESS"
            not in elemento):
        lista_file_movies_finale.append(elemento)
lista_file_movies_finale = sorted(lista_file_movies_finale)

lista_file_users_finale = []
for elemento in lista_file_users:
    if (".part" not in elemento and "._SUCCESS" not in elemento and "_SUCCESS"
            not in elemento):
        lista_file_users_finale.append(elemento)
lista_file_users_finale = sorted(lista_file_users_finale)
print("File non rilevanti eliminati.")

# Salviamo quindi i file di testo restanti generati da Spark sul nostro
# database. In questo caso vogliamo salvare molti documenti in un unica
# collection. Poiché abbiamo molti file nella cartella, iteriamo sui suoi
# elementi e ne apriamo uno alla volta. Dopodiché, poiché ogni riga è un
# file Json, iteriamo sulle righe stesse.
print("Inizio processamento file.")
lista_documenti_ratings = []
count = 0
for file in lista_file_ratings_finale:
    with open(f"{percorso}/Ratings_json/{file}") as f:
        for line in f:
            lista_documenti_ratings.append(json.loads(line))
    count += 1
    print(f"{count} file ratings su {len(lista_file_ratings_finale)} processati.")

lista_documenti_movies = []
count = 0
for file in lista_file_movies_finale:
    with open(f"{percorso}/Movies_json/{file}") as f:
        for line in f:
            lista_documenti_movies.append(json.loads(line))
    count += 1
    print(f"{count} file movies su {len(lista_file_movies_finale)} processati.")

lista_documenti_users = []
count = 0
for file in lista_file_users_finale:
    with open(f"{percorso}/Users_json/{file}") as f:
        for line in f:
            lista_documenti_users.append(json.loads(line))
    count += 1
    print(f"{count} file users su {len(lista_file_users_finale)} processati.")
print("Tutti i file sono stati processati.")

# A questo punto possiamo finalmente creare il database su MongoDB e le varie collezioni.
# Chiamiamo il database 'Movielens'. Poiché ci sono diversi documenti all'interno
# di ciascun file, usiamo insert_many.
print("Creazione dataset in corso. Attendere.")
ora_prima = datetime.datetime.now()
client["Movielens"]["Ratings"].insert_many(lista_documenti_ratings)
print("Collezione Ratings creata.")
client["Movielens"]["Movies"].insert_many(lista_documenti_movies)
print("Collezione Movies creata")
client["Movielens"]["Users"].insert_many(lista_documenti_users)
print("Collezione Users creata")
ora_dopo = datetime.datetime.now()
print("Creazione dataset su MongoDB completata. Tempo impiegato:"
      f"{ora_dopo - ora_prima}")

# Calcoliamo quindi il tempo di esecuzione della prima fase.
ora_fine_P1 = datetime.datetime.now()
tempo_P1 = ora_fine_P1 - ora_inizio_P1

for i in range(0, 4):
    print("*" * 30)
    print()
    if i == 1:
        print(f"FASE 1 COMPLETATA IN {tempo_P1}. INIZIO FASE 2.")
        print()

"""
SECONDA FASE: Implementazione dell'algoritmo.
"""
# Come per la prima fase, prendiamo il tempo di esecuzione.
ora_inizio_P2 = datetime.datetime.now()

print("Inizio implementazione algoritmo.")
# Poiché gli autori dell'articolo utilizzano la libreria MLlib, in essa la
# funzione ALS si aspetta in input un RDD di tipo Rating. Essa rappresenta
# una tupla formata nel seguente modo: (user, product, rating). Quindi,
# procediamo alla trasformazione.
dRating5 = dRating2.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
print("Trasformazione in oggetti Rating effettuata.")

# A questo punto, per poter valutare le prestazioni del modello che noi
# addestreremo, è bene dividere i dati in due insiemi; con un insieme
# addestreremo il modello, mentre con un altro insieme lo valuteremo. Il
# vantaggio fondamentale è che il modello non ha mai visto in fase di
# addestramento l'insieme che utilizzeremo per la valutazione, cosa che
# simula la realtà di usare il modello su dati che esso non ha mai visto.
# Chiameremo l'insieme con cui il modello verrà addestrato dTrain e quello
# con cui verrà valutato dTest. Per suddividere in modo casuale useremo
# randomSplit.
dTrain, dTest = dRating5.randomSplit([0.7, 0.3], seed=42)
print("Suddivisione del data set in training set e test set completata.")

# Poiché la suddivisione è casuale, andiamo a ordinare i valori.
dTest2 = dTest.sortBy(lambda x: (x.user, x.product))

# Possiamo finalmente addestrare il modello. Gli iperparametri selezionati sono
# gli stessi che gli autori hanno usato nell'articolo.
print("Addestrando il modello...")
model = ALS.train(dTrain, rank=10, iterations=5, lambda_=0.05)
print("Addestramento del modello completato.")

# Una volta addestrato il modello possiamo effettuare le previsioni. Prima di
# fare ciò, andiamo a togliere la colonna Rating dall'insieme dTest in quanto
# la funzione predictAll accetta coppie di user - prodotto.
dTest3 = dTest2.map(lambda x: (x.user, x.product))

# Possiamo ora usare il modello per effettuare le previsioni della colonna
# Rating sull'insieme di testing. Quello che abbiamo fatto in sostanza è
# stato rimuovere le vere valutazione e sostituirle con quelle predette dal modello.
dPredict = model.predictAll(dTest3)
print(f"Previsioni effettuate.")

# Le previsioni sono anch'esse un oggetto rating. Le trasformiamo quindi in una tupla.
dPredict2 = dPredict.map(lambda x: ((x[0], x[1]), x[2]))

# Le ordiniamo poiché con la funzione dPredict l'ordine di partenza non
# è rispettato.
dPredict3 = dPredict2.sortBy(lambda x: (x[0][0], x[0][1]))
# Appunto: la riga sopra mostra un messaggio di avviso. Il codice però funziona regolarmente.

print(f"Alcune delle previsioni effettuate sono: {dPredict3.take(5)}")

# A questo punto possiamo trasformare anche le valutazioni reali, le quali sono
# anche'esse un oggetto rating. Quindi, le riportiamo sotto forma di tupla.
dReali = dTest2.map(lambda x: ((x[0], x[1]), x[2]))

# I passaggi precedenti ci permettono di unire le previsioni effettuate tramite
# il modello con le previsioni reali.
dUnione = dPredict3.join(dReali)

# Le ordiniamo.
dUnione2 = dUnione.sortBy(lambda x: (x[0][0], x[0][1]))

print(f"Alcune valutazioni predette affiancate a quelle reali sono: {dUnione.take(5)}")

# Come possiamo notare, abbiamo ottenuto dai passaggi precedente un serie di tuple.
# In particolare ogni riga ha due tuple, una con all'interno gli Id dei film e degli
# utenti e un'altra con la valutazione predetta e reale. A noi interessa solo la seconda.
dUnione3 = dUnione2.map(lambda x: x[1])

# Possiamo ora usare la funzione RegressionMetrics di MLlib per poter creare un oggetto
# che possiamo usare per calcolare varie metriche, come per esempio l'RMSE.
dMetrics = RegressionMetrics(dUnione3)

# Calcoliamo quindi l'RMSE.
dRmse = dMetrics.rootMeanSquaredError
print(f"L'RMSE calcolato sul test set è pari a: {dRmse}")
print("Algoritmo implementato correttamente.")

# Abbiamo terminato la seconda fase, quindi calcoliamo il suo tempo di esecuzione.
ora_fine_P2 = datetime.datetime.now()
tempo_P2 = ora_fine_P2 - ora_inizio_P2
for i in range(0, 4):
    print("*" * 30)
    print()
    if i == 1:
        print(f"FASE 2 COMPLETATA IN {tempo_P2}. INIZIO FASE 3.")
        print()

"""
TERZA FASE: Produzione raccomandazioni per utente e caricamento dei risultati 
prodotti su MongoDB.
"""
# Prima di cominciare, prendiamo il tempo anche della terza fase.
ora_inizio_P3 = datetime.datetime.now()

# A questo punto possiamo finalmente produrre le raccomandazioni per tutti gli utenti.
# Facciamo questo con la funzione recommendProductsForUsers(), dove, tra le parentesi
# vanno specificate il numero di raccomandazioni che vogliamo fare per ogni utente.
# Faremo 5 raccomandazioni per utente.
print("Effettuando le raccomandazioni per ogni utente...")
dRecommendations = model.recommendProductsForUsers(5)
print("Raccomandazioni completate.")

# Diamo un sguardo a come sono strutturate.
print(f"Alcune raccomandazioni sono le seguenti: {dRecommendations.take(5)}")

# Come possiamo notare esse sono rappresentate da collezioni. In particolare abbiamo
# l'Id utente come chiave e poi una tupla contente oggetti rating, uno per raccomandazione.
# Poiché l'Id dell'utente è contenuto anche nell'oggetto rating, prendiamo solo quest'ultimo.
print("Inizio trasformazioni recommendations.")
dRecommendations1 = dRecommendations.flatMap(lambda x: x[1])

# Trasformiamo quindi l'oggetto rating in una tupla dove il secondo elemento è un
# dizionario.
dRecommendations2 = dRecommendations1.map(lambda x: (x.user, ({"MovieId": x.product, "Rating": x.rating})))

# Raggruppiamo tutti gli utenti che hanno lo stesso UserId.
dRecommendations3 = dRecommendations2.groupByKey().mapValues(list)

# A questo punto creiamo un dizionario vero e proprio in quanto successivamente queste
# raccomandazioni verranno salvate su MongoDB. In particolare, il primo elemento sarà
# lo UserId, mentre come secondo elemento avremo a nostra volta un dizionario contenente
# il MovieId e il Rating. In questo caso il Rating è quello predetto.
dRecommendations4 = dRecommendations3.map(lambda x: {"UserId": x[0], "FilmRecommended": x[1]})

# Ordiniamo il dizionario per UserId.
dRecommendations5 = dRecommendations4.sortBy(lambda x: x["UserId"])

# Diamo adesso uno sguardo a come appaiono le raccomandazioni prodotte.
print("Fine trasformazioni recommendations.")
print(f"Alcune raccomandazioni sono le seguenti: {dRecommendations5.take(5)}")

# A questo punto vogliamo andare a salvare le raccomandazioni prodotte sul database
# MongoDB. Per farlo, come fu per la prima fase, andiamo a usare una funzione del
# pacchetto json, così da preparare il tutto al salvataggio (ricordiamo che MongoDB
# si aspetta dei documenti in formato json).
dRecommendations6 = dRecommendations5.map(lambda x: json.dumps(x))

# Salviamo quindi le RDD sul nostro computer.
dRecommendations6.saveAsTextFile(percorso + "/Recommendations_json")
print("File recommendations salvati correttamente.")

# Guardiamo alle cartelle salvate.
lista_file_recommendations = os.listdir(percorso + "/Recommendations_json")
print(f"L'elenco dei file salvati per le recommendations è il seguente: {lista_file_recommendations}")

# Prendiamo solo i file effettivamente utili all'interno della cartella.
print("Eliminazione file inutili...")
lista_file_recommendations_finale = []
for elemento in lista_file_recommendations:
    if (".part" not in elemento and "._SUCCESS" not in elemento and
       "_SUCCESS" not in elemento):
        lista_file_recommendations_finale.append(elemento)
lista_file_recommendations_finale = sorted(lista_file_recommendations_finale)
print("File non rilevanti eliminati.")

# Salviamo i file su MongoDB
print("Inizio processamento file.")
lista_documenti_recommendations = []
count = 0
for file in lista_file_recommendations_finale:
    with open(f"{percorso}/Recommendations_json/{file}") as f:
        for line in f:
            lista_documenti_recommendations.append(json.loads(line))
    count += 1
    print(f"{count} file recommendations su {len(lista_file_recommendations_finale)} processati.")
print("Tutti i file sono stati processati.")

# Carichiamo finalmente le raccomandazioni sul nostro database.
client["Movielens"]["Recommendations"].insert_many(lista_documenti_recommendations)
print("Collezione Recommendations creata.")

# Calcoliamo ora il tempo di esecuzione della terza fase.
ora_fine_P3 = datetime.datetime.now()
tempo_P3 = ora_fine_P3 - ora_inizio_P3
for i in range(0, 4):
    print("*" * 30)
    print()
    if i == 1:
        print(f"FASE 3 COMPLETATA IN {tempo_P3}.")
        print()

# Calcoliamo il tempo finale.
tempo_finale = tempo_P1 + tempo_P2 + tempo_P3
for i in range(0, 4):
    print("*" * 30)
    print()
    if i == 1:
        print(f"TUTTE LE FASI SONO STATE COMPLETATE. TEMPO DI ESECUZIONE TOTALE: {tempo_finale}")
        print()

"""
QUERY UTILI PER INTERROGARE MONGODB
"""
print("Il codice principale è concluso. Effettuare delle query di esempio per vederne i "
      "risultati?")
risposta = int(input("Digitare 1 se si vogliono vedere delle query di esempio, "
                     "digitare 0 per concludere l'esecuzione."))
if risposta == 1:
    # A questo punto come ultimissima cosa possiamo mostrare delle semplici
    # query che possiamo utilizzare per interrogare il nostro database.

    # PRIMA QUERY
    # Supponiamo che siamo interessati a cercare delle informazioni su un certo
    # utente.
    while True:
        UserId = int(input("Inserire il codice dell'utente che si vuole cercare. "
                           "Il codice è un numero intero positivo."))
        collection1 = client["Movielens"]["Users"]
        query1 = collection1.find_one({"UserId": UserId}, {"_id": 0})
        if query1:
            print(f"Il profilo dell'utente {UserId} è il seguente: {query1}")
            break
        else:
            print(f"Nessun utente trovato con id {UserId}.")

    # SECONDA QUERY
    # A questo punto possiamo notare che l'utente x ha visto y film e sia
    # indeciso su cosa guardare.
    # Possiamo quindi eseguire una seconda query, la quale trova le raccomandazioni
    # per l'utente selezionato.
    while True:
        UserId2 = int(input("Cercare raccomandazioni per lo stesso utente cercato "
                            "prima? Digitare 1 per indicare 'sì' e 0 per indicare 'no'."))
        if UserId2 == 1:
            collection2 = client["Movielens"]["Recommendations"]
            query2 = collection2.find_one({"UserId": UserId}, {"_id": 0})
            print(f"Le raccomandazioni per l'utente {UserId} sono: {query2}")
            break
        else:
            if UserId2 == 0:
                UserId3 = int(input("Digitare il nuovo id utente per il quale si vogliono cercare "
                                    "i film raccomandati. L'id è un numero intero positivo."))
                collection2 = client["Movielens"]["Recommendations"]
                query2 = collection2.find_one({"UserId": UserId3}, {"_id": 0})
                if query2:
                    print(f"Le raccomandazioni per l'utente {UserId3} sono: {query2}")
                    break
                else:
                    print(f"Nessun utente trovato con id {UserId3}")
            else:
                print("E' stato inserito un valore non accettato. Rispondere 1 per indicare "
                      "'sì', 0 per indicare 'no'")

    # TERZA QUERY
    # A questo punto notiamo che la query precedente ci ha raccomandato 5 film.
    # Possiamo quindi scrivere una query per cercare informazioni su quei film.
    while True:
        MovieId = int(input("Inserire l'id del film del quale si vogliono cercare informazioni."))
        collection3 = client["Movielens"]["Movies"]
        query3 = collection3.find_one({"MovieId": MovieId}, {"_id": 0})
        if query3:
            print(f"Le informazioni relative al film con id {MovieId} sono le seguenti: {query3}")
            break
        else:
            print(f"Nessun film trovato con id {MovieId}")

    # QUARTA QUERY
    # Infine, dopo aver ottenuto informazioni su un particolare film, possibilmente uno
    # di quelli raccomandati, magari siamo interessati a sapere cosa ne pensano gli altri
    # utenti. Possiamo quindi fare una query che ci restituisca le votazioni degli utenti
    # a quel film, espandendola per renderla anche generica, così come fatto per la seconda
    # query.
    while True:
        MovieId2 = int(input("Cercare votazioni per lo stesso film cercato "
                             "prima? Digitare 1 per indicare 'sì' e 0 per indicare 'no'."))
        if MovieId2 == 1:
            collection4 = client["Movielens"]["Ratings"]
            query4 = collection4.find({"MovieId": MovieId}, {"_id": 0, "MovieId": 0}).limit(10)
            query4 = list(query4)
            if query4:
                for doc in query4:
                    print(f"Le valutazioni per il film {MovieId} sono: {doc}")
                break
        else:
            if MovieId2 == 0:
                MovieId3 = int(input("Digitare l'id del film per il quale si vogliono cercare "
                                     "le valutazioni degli utenti. L'id è un numero intero positivo."))
                collection4 = client["Movielens"]["Ratings"]
                query4 = collection4.find({"MovieId": MovieId3}, {"_id": 0, "MovieId": 0}).limit(10)
                query4 = list(query4)
                if query4:
                    for doc in query4:
                        print(f"Le valutazioni per il film {MovieId3} sono: {doc}")
                    break
                else:
                    print(f"Nessun film trovato con id {MovieId3}")
            else:
                print("E' stato inserito un valore non accettato. Rispondere 1 per indicare "
                      "'sì', 0 per indicare 'no'")
else:
    print("Fine.")
