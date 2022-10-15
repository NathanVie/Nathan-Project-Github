# -*- coding: utf-8 -*-
"""
Nathan VIEIRA
Maxime SOKOL
Ilann SARR DIARRA
Mathis TIBERGHIEN
"""

from IPython import get_ipython

get_ipython().magic('reset -sf')

import numpy as np
import sys
from copy import copy
import time
import string


lignes = 15
colonnes = 15

board = np.zeros((lignes,colonnes))

inf = 9999999999
n_inf = -9999999999
alphabet_string = string.ascii_uppercase
alphabet_list = list(alphabet_string)

def Affichage():
    print("    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15")
    x=1
    for i in range(0, lignes):

        print(alphabet_list[i], end = " | ")

        for j in range(0, colonnes):
            if board[i, j] == 0:
                sys.stdout.write('_| ')
            elif board[i, j] == 1:
                sys.stdout.write('X| ')
            else:
                sys.stdout.write('O| ')
        x+=1
        print ('')

def Heuristique(lignes, colonnes):
    PLigne, PColonne, PDiagonale, PDiagonaleInv, ListeF, n = [], [], [], [], [], 0
    
    for ligne in range(15): #Automatisme pour trouver les solutions possibles
        for colonne in range(15):
            if(colonne+4 <= 14):
                Posib = []
                for i in range(0,5):          #on ajoute TOUTES les possibilités de lignes
                    Posib.append(n+i)
                PLigne.append(Posib)
                n += 1
        n += 4
    
    for colonne in range(15):
        n = 0
        for ligne in range(15):
            if(ligne+4 <= 14):
                Posib = []
                for i in range(0,61,15):          #on ajoute TOUTES les possibilités de colonnes
                    Posib.append(n+colonne+i)
                PColonne.append(Posib)
                n += 15
    
    suivb = 0
    suivh = 0
    
    for ligne in range(1, 12):
        sol = 11 - ligne
        add = 0 
        for y in range(sol+1):
            Posibb, Posibh = [], []
            for i in range(0,65,16):
                Posibb.append(suivb+add+i)          #on ajoute TOUTES les possibilités de victoire diagonale
                Posibh.append(suivh+add+i)
            PDiagonale.append(Posibb) 
            if(Posibh not in PDiagonale):
                PDiagonale.append(Posibh)    
            add += 16
        suivb += 15
        suivh += 1    
    
    suivb = 14
    suivh = 14
    
    for ligne in range(1, 12):
        sol = 11 - ligne
        add = 0
        for y in range(sol+1):
            Posibb2, Posibh2 = [], []
            for i in range(0,57,14):
                Posibb2.append(suivb+add+i)          #on ajoute TOUTES les possibilités de victoire diagonale inversée
                Posibh2.append(suivh+add+i)
            PDiagonaleInv.append(Posibb2) 
            if(Posibh2 not in PDiagonaleInv):
                PDiagonaleInv.append(Posibh2)    
            add += 14
        suivb += 15
        suivh -= 1
    
    ListeF = PLigne+PColonne+PDiagonale+PDiagonaleInv               #liste de listes gagnantes
    
    tabGagnant=np.array(ListeF)                       #tableau gagnant = toutes les listes gagnantes en 1 tableau
    tabHeuristique = np.zeros((lignes+1,colonnes+1))    #table heuristique = matrice 16/16 de zeros
    nbPosGagnantes = len(ListeF)                         #nombre de positions gagnantes : longueur de la liste positions gagnantes
    
    for i in range(0, lignes+1):                        #pour i allant de 0 a 16    (donc 0 a 15 compris)
        tabHeuristique[i, 0] = 10**i                    #colonne 1 = valeurs allant de 1 à 10000000000
        tabHeuristique[0, i] = -10**i                   #ligne 1 = valeurs allant de -1 à -10000000000
    
    return tabGagnant, nbPosGagnantes, tabHeuristique


def Evaluation(etat, tabGagnant, nbPosGagnantes, tabHeuristique):

    etatCopy = copy(etat.ravel())                   #on aplatit le tableau en une liste pour avoir le tableau sous forme d'une liste 1D
    heuristique = 0                                 #on initialise l'heuristique a 0
    
    for i in range(0, nbPosGagnantes):              #0 à nombre de positions gagnantes
        maxp = 0                                    #on initialise la valeur max
        minp = 0                                    #on initialise la valeur min
        for j in range(0, 5):                       #range de la ligne = victoire 
            if etatCopy[tabGagnant[i,j]] == 2:      #si la valeur de etat a l'index de la valeur du tableau gagnant vaut 2
                maxp += 1                           #on incremente max
            elif etatCopy[tabGagnant[i,j]] == 1:    #si la valeur de etat a l'index de la valeur du tableau gagnant vaut 1
                minp += 1                           #on incremente max
        
        heuristique += tabHeuristique[maxp][minp]     
    return heuristique                              #on retourne l'heuristique

# etat = etat actuel du tableau
# alpha
# beta
# maximizing qui va déterminer si l’étape doit retourner le coup à l’heuristique la plus haute
#   (tour de l’IA) ou le coup à l’heuristique la plus basse (tour de l’utilisateur).
# depth qui donne la limite de profondeur des coups à comparer, pour éviter que le
#   programmme mette trop de temps à s’exécuter
# maxp valeur maximale du meilleur coup
# minp valeur minimale du meilleur coup


def Minimax(etat, alpha, beta, maximizing, depth, maxp, minp):
    tabGagnant, nbPosGagnantes, tabHeuristique = Heuristique(lignes, colonnes)
    heuristique = Evaluation(etat, tabGagnant, nbPosGagnantes, tabHeuristique)

    if depth == 0:                                       #si limite = 0 alors on retourne
        return heuristique, etat                           #retourne heuristique et etat du tableau
    
    lignesRest, colonnesRest = np.where(etat == 0)        #lignes restantes, colonnes restantes la ou l'etat vaut encore 0
    returnEtat = copy(etat)                              #on save l'etat dans un autre espace de stockage
    
    if lignesRest.shape[0] == 0:                          #s'il ne reste plus aucune ligne avec etat = 0 retourne heuristique et 
        return heuristique, returnEtat                   #retourne heuristique et etat du tableau
        
    if maximizing == True:                              #si max
        
        eval = n_inf
        for i in range(0, lignesRest.shape[0]):                          #pour i allant de 0 à la longueur des lignes qu'il reste
            nextEtat = copy(etat)                                    #on copie l'etat du tableau
            nextEtat[lignesRest[i], colonnesRest[i]] = maxp              #le prochain tableau sera composé de maxp aux indexs des valeurs restantes 
            Neval, Netat = Minimax(nextEtat, alpha, beta, False, depth-1, maxp, minp)   #evaluation et nextEtat (attention on passe à maximizing = False pour changer le status)
            
            if Neval > eval:                                     #si evaluation > -infini 
                eval = Neval                                     #evaluation = evaluation obtenue
                returnEtat = copy(nextEtat)                          #et on recopie le Etat
                
            if eval > alpha :                                       #si evaluation > alpha alors on garde l'evaluation d'origine
                alpha = eval                                        #et on set alpha = -infini
                
            if alpha >= beta :                                         #si alpha >= beta
                break;                                                 #stop
    
        return eval, returnEtat

    else:                                                     #si min
        eval = inf                                                  
        for i in range(0, lignesRest.shape[0]):                           #pour i allant de 0 à la longueur des lignes qu'il reste 
            nextEtat = copy(etat)                                    #on copie l'etat du tableau
            nextEtat[lignesRest[i],colonnesRest[i]] = minp               #le prochain tableau sera composé de minp aux indexs des valeurs restantes
            Neval, Netat = Minimax(nextEtat, alpha, beta, True, depth-1, maxp, minp)    #heuristique et nextEtat (attention maximizing = True)
            
            if Neval < eval:                                     #si heuristique < +infini 
                eval = Neval                                     #heuristique = heuristique obtenue
                returnEtat = copy(nextEtat)                          #et on recopie le Etat
                
            if eval < beta :                                        #si heuristique < beta alors
                beta = eval                                         #alpha = +infini
                
            if alpha >= beta :                                         #si alpha >= beta
                break;                                                 #stop
                
        return eval, returnEtat                                    #retourne heuristique et etat du tableau

def Verification(etat):
    etatCopie = copy(etat)                    #on duplique l'etat de l'etat du tableau
    tabGagnant, nbPosGagnantes, tabHeuristique = Heuristique(lignes, colonnes)
    heuristique = Evaluation(etatCopie, tabGagnant, nbPosGagnantes, tabHeuristique)          #calcul de l'heuristique du tableau etat poura voir condition de victoire ou de defaite
    
    if abs(heuristique) >= 100000 :              #si heuristique >100000 victoire 
        return 1
    else :                                      #sinon coinue de jouer
        return -1
    
def Conditions():

                                                                      
    c = int(input('Colonne = '))
    r1 = str(input('Ligne = '))
    
    r1 = r1.upper()
    if r1.isalpha()==False:
        print("Veuillez rentrer une index de ligne valide")
        return Conditions()
    r = alphabet_list.index(r1) + 1
    if (c<1) or (isinstance(c,int) == False) or (c>15):
            print("veuillez écrire un nombre compris dans le tableau")
            return Conditions()
    
    
    
    if (isinstance(r, int) == False) or (r < 1) or (r > 15):
        print("veuillez écrire une lettre comprise dans le tableau")
        return Conditions()
    
       
    
       
    if board[r-1, c-1] == 1 or board[r-1,c-1] == 2 :
        print("cette position est déjà occupée, saisissez en une nouvelle")
        return Conditions()
       
  
    return (c,r)

def Condi():
    ans = int(input('User ou IA (1/2) : '))
    if ans == 1 or ans== 2:
        return ans
    else:
        print("Veuillez saisir les nombres 1 ou 2")
        return Condi()
    
def Main():
    ans = Condi()
    val = 0
    global board                                #on initialise board a zeros dans tout le code
    #Affichage()
    
    
    if ans==2:              #Si l'IA commence, elle joue en H8
        board[7,7] = 2
        Affichage()
        
        print('\nC\'est le tour de l\'utilisateur')
        (c,r)=Conditions()
        board[r-1,c-1] = 1                  #on remplace 0 par 1 pour le tour du joueur humain
        val = Verification(board)
        
        if board [7,3] == 1:        #Puis elle joue dans une case inoccupée n'était pas dans le carré 7x7 autour du premier pion
            board [7,2] = 2
            Affichage()
        else:
            board [7,3] = 2
            Affichage()
        
        ans = 1  
    
    else:
        Affichage()
                                                #on affiche le tableau
    for turn in range(0, lignes*colonnes):      #turn limite a la taille du tableau
        if (turn+ans) % 2 == 1: 
            print('\nC\'est le tour de l\'utilisateur')
            (c,r)=Conditions()
            board[r-1,c-1] = 1                  #on remplace 0 par 1 pour le tour du joueur humain
            Affichage()                         #on reaffiche le tableau
            val = Verification(board)         #on verifie le tableau - retourne 1 si le jeu est fini / retourne -1 si le jeu continue 
        
            if val == 1:                      #si val = 1 alors fin du jeu
                print ('L\'utilisateur a gagne !')
                sys.exit()                      
            print ('\n')
                     
        else: 
            print('C\'est le tour de l\'IA')
            print ('\n')
            start_time = time.time()            #compteur pour avoir le temps de l'exécution de l'IA
            etat = copy(board)                 #on copie le tableau
            val, nextEtat = Minimax(etat, n_inf, inf, True, 1, 2, 1)   #on fait un MinMax pour obtenir val (1 ou -1) et nextEtat / on paramètre la profondeur à 1                             
            board = copy(nextEtat)             #copie l'état de sortie du MinMax
            Affichage()                         #on affiche le tableau
            print('\n')
            print("Temps d'execution de l'IA : ", int(round((time.time() - start_time ), 0)),"secondes")    #print le temps d'exécution
            val = Verification(board)   
            
            if val == 1:                      #si val = 1 alors fin du jeu
                print ('L\'IA a gagne !')
                sys.exit()
        
    print ('Egalité')                           #si on a rien

Main()
