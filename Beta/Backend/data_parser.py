#!/usr/bin/env python3
'''
AITrain data parser
'''
# -*- encoding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from datetime import datetime

import numpy as np
import pandas as pd

#define
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
SESSION_PATH = 'raw_data/sessions'
WELLNESS_PATH = 'raw_data/wellness'
FORMAT = '%Y-%m-%d'

def from_str_to_date(stringa):
    timestr = stringa.split(' ')[0]
    time = datetime.strptime(timestr, FORMAT)
    return time

def import_sessions():
    #importo csv_session
    df_session = pd.DataFrame()
    df_exercises = pd.DataFrame()
    list_files = os.listdir(os.path.join(FOLDER_PATH, SESSION_PATH))

    lista_colonne_utili_analisi_esercizi = ['splitName',
                                            'workload',
                                            'perceivedWorkload',
                                            'metabolicWorkload',
                                            'mechanicalWorkload',
                                            'kinematicWorkload',
                                            'cardioWorkload']

    #list_files = list_files[:6] #per debug
    i = 0 #brutto ma efficace
    for file in list_files:
        print('sto analizzando il file: ', file)

        #importo csv data singola
        tempdf = pd.DataFrame(pd.read_csv(os.path.join(FOLDER_PATH, SESSION_PATH, file)))

        #sistemo formato date
        tempdf['date'] = tempdf['date'].map(from_str_to_date)

        #giocatori allenatisi in quella sessione
        player_list = list(tempdf['playerName'].unique())

        #per ogni giocatore produco il record di session
        for curr_player in player_list:
            print('        sto analizzando il giocatore: ', curr_player)

            #filtra df in base a player name
            playerdf = tempdf[tempdf['playerName'].isin([curr_player])]

            #Tutti i dati della session
            sessiondf = playerdf[playerdf['splitName'].isin(['Session', 'session'])].drop('splitName', axis=1)
            if sessiondf.empty:
                print('                 Eccezione game??????????????????????????????????')
                sessiondf = playerdf[playerdf['splitName'].isin(['game', 'Game'])].drop('splitName', axis=1)

            if sessiondf.empty:
                print('##################Eccezione warmup??????????????????????????????????')
                sessiondf = playerdf[playerdf['splitName'].isin(['Warm-up'])].drop('splitName', axis=1)

            #a cui aggiungo la lista degli esercizi
            exercises = list(playerdf['splitName'].unique())

            sessiondf['exercises'] = [exercises] #necessaria la doppia lista per avere un singolo elemento

            #workloads su singoli esercizi
            #al momento è inutile perchè non calcolano wl al singolo esercizio
            esercizidf = playerdf[lista_colonne_utili_analisi_esercizi]

            if i == 0:
                df_session = sessiondf.copy()
                df_exercises = esercizidf.copy()
                i += 1
            else:
                df_session = df_session.append(sessiondf, ignore_index=True)
                df_exercises = df_exercises.append(esercizidf, ignore_index=True)

    da_tenere = [
        'date', 'playerName', 'duration', 'rpe', 'workload',
        'complete',
        'perceivedWorkload', 'metabolicWorkload', 'mechanicalWorkload',
        'kinematicWorkload', 'cardioWorkload', 'intensity', 'PLAYER NAME',
        'Player Load', 'Player Name',
        'Work Ratio', 'gdType', 'rpeTl', 'exercises'
    ]
    df_session = df_session[da_tenere]
    # df con tutte le session

    return df_session, df_exercises

def import_wellness():
    #lista csv
    list_files = os.listdir(os.path.join(FOLDER_PATH, WELLNESS_PATH))

    i = 0
    for file in list_files:
        print('sto analizzando il file: ', file)
        #importo csv
        df_tempwellness = pd.DataFrame(pd.read_csv(os.path.join(FOLDER_PATH, WELLNESS_PATH, file)))

        #formattazione data corretta
        df_tempwellness['date'] = df_tempwellness['date'].map(from_str_to_date)

        #join session con il wellness
        #Oss: outer vuol dire che faccio l'unione insemistica dei due db
        #(quindi se il giocatore si allena e non fa questionario o viceversa, si aggiungono NaN)
        #Problema: se faccio merge e la colonna esiste mi crea una colonna nuova ogni volta
        #Sol: fare outer su una data e appendere al risultato
        #probl: anche adesso ripetizioni
        #Sol: creare un df totale di wellness e mergare quello

        if i == 0:
            wellness_df = df_tempwellness
            i += 1
        else:
            wellness_df = wellness_df.append(df_tempwellness, ignore_index=True)

    return wellness_df

def recupera_esercizi(riga_df, df_session_player_temp):
    date = riga_df[0] #data della riga

     #vado a cercare i vari campi esercizi associati alla data
    list_execises_date = list(df_session_player_temp[df_session_player_temp['date'].isin([date])]['exercises'])

    #è una lista di liste, quindi devo ricompattare

    lista_esercizi = []
    for x in list_execises_date:
        lista_esercizi = lista_esercizi + x
    riga_df['exercises'] = lista_esercizi


    return riga_df

def import_data():
    #df con tutti i wellness
    df_session, df_exercises = import_sessions()
    wellness_df = import_wellness()

    pd.value_counts(df_session['playerName'], dropna=False)

    start_date = from_str_to_date('2017-06-12')
    end_date = from_str_to_date('2018-04-10')

    player_list = list(df_session['playerName'].unique())

    #funzione che sistema i doppi allenamenti :)

    for player in player_list:
        print(player)
        print('')
        df_session_player_temp = df_session[df_session['playerName'].isin([player])]

        df_session_player = df_session_player_temp.groupby('date').sum() #join delle righe con la stessa data sommando
        #Oss: perdo la colonna playername e exercises, da recuperare
        #perdo anche completed, non mi interessa, non so perchè la tenevo

        df_session_player = df_session_player.reset_index() #non so se sia utile ma lo faccio

        temp_df = df_session_player.apply(recupera_esercizi, args=(df_session_player_temp,), axis=1)

        #metto la data come indice
        temp_df = temp_df.set_index('date')

        #aggiungo tutte le date mancanti dalla iniziale, mettendo 0 al posto di na => (non allenamenti)
        new_index = pd.Index(pd.date_range(start_date, end_date).tolist())
        temp_df1 = pd.DataFrame(temp_df, index=new_index).fillna(0)

        #aggiungo la colonna playerName
        temp_df1['playerName'] = player
        #filtro per giocatore
        df_wellness_player = wellness_df[wellness_df['playerName'].isin([player])]

        #drop della colonna playerName
        df_wellness_player = df_wellness_player.drop(['playerName'], axis=1)

        #metto la data come indice
        df_wellness_player = df_wellness_player.set_index('date')

        #cerco eventuali date doppie, fanno casino con gli indici
        list_index_duplicated = list(set(df_wellness_player[df_wellness_player.index.duplicated(keep=False)].index))
        for date2 in list_index_duplicated:
            print('DATA CON PIU OSSERVAZIONI 1')
            print(df_wellness_player.loc[[date2]])

        #droppo i duplicati
        #Oss:quando resetto un indice la colonna che prima era indice diventa 'index'
        if list_index_duplicated != []:
            da_buttare = df_wellness_player.index.duplicated(keep=False)

            #elimino quelli con campi uguali a 0 o 1
            result = np.array([False for x in da_buttare]) #tutti falsi, nessuno da buttare
            for column in list(df_wellness_player.columns.values):
                result = np.logical_or(df_wellness_player[column] == 0, result) #con l'or vanno a true(da buttare) quelli con uno 0
                result = np.logical_or(result, df_wellness_player[column] == 1) #con l'or vanno a true(da buttare) quelli con un 1

            da_buttare = np.logical_and(da_buttare, result) # da buttare sono quelli duplicati(contenuti in da buttare) e che hanno uno 0 o 1

            #mantengo quelli da non buttare
            df_wellness_player = df_wellness_player[~da_buttare]

            for date3 in list_index_duplicated:
                print('DATA CON PIU OSSERVAZIONI 2')
                print(df_wellness_player.loc[[date3]])

            #elimino i duplicati rimasti, tenendo il primo
            df_wellness_player = df_wellness_player[~df_wellness_player.index.duplicated(keep='first')]

        #aggiungo tutte le date mancanti dalla iniziale, mettendo 0 al posto di na => (non allenamenti)
        new_index = pd.Index(pd.date_range(start_date, end_date).tolist())

        df_wellness_player = pd.DataFrame(df_wellness_player, index=new_index)

        ##########################################################################
        #questo sporca enormemente i dati per il DNN, preferirei commentarlo in futuro
        #magari mettere 5 5 5 5 5 ? spesso i buchi sono dove non c'è allenamento
        df_wellness_player = df_wellness_player.fillna(df_wellness_player.mean())
        ##########################################################################

        df_result = pd.merge(temp_df1.reset_index(), df_wellness_player.reset_index(), on=['index'])
        #Oss: reset_index mi cfa tornare date a essere una colonna normale ma la chiama index
        #rinomino la colonna index in date
        df_result.rename(columns={'index': 'date'}, inplace=True)

        globals()[player] = df_result

    output_data = []

    for player in player_list:
        output_data.append(globals()[player])

    return output_data

def main():
    import_data()

if __name__ == '__main__':
    main()