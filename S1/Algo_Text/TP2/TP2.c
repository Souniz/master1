#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include <stdbool.h>
int recherche_naive_sans_boucle_rapide(char* mot,char* texte)
{
    int non_match=0;
    int nb_occur=0;
    for (int i=0;i<=strlen(texte)-strlen(mot);i++)
    {
        for (int j=0;j<strlen(mot);j++)
        {
            if(texte[i+j]!=mot[j]){
                non_match=1;
                break;
            } 
        } 
        if(non_match==0)
        {
            nb_occur+=1;
        }
        non_match=0;
    }
    return nb_occur;
}
//-----------------------------------------------------------
int recherche_naive_avec_boucle_rapide(char* mot,char* texte)
{
    int non_match=0;
    int nb_occur=0;
    char firts_lettre=mot[0];
    for (int i=0;i<=strlen(texte)-strlen(mot);i++){

        if (firts_lettre==texte[i]){

            for (int j=0;j<strlen(mot);j++){
                if(texte[i+j]!=mot[j]){
                    non_match=1;
                    break;
                } 
            } 
            if(non_match==0)
            {
                nb_occur+=1;
            }  
        
        }
        
        non_match=0;
    }
    return nb_occur;
}
//----------------------sentinelle------------------
int recherche_naive_avec_boucle_rapide_sentinelle(char* mot,char* texte){  
            
    texte = (char*)realloc(texte, (strlen(texte) + strlen(mot) + 1) * sizeof(char));
    strcat(texte, mot);
    int non_match=0;
    int nb_occur=0;
    int taille=strlen(texte)-strlen(mot);
    int i=0;
    int j;
    while(true)
    {
        j=0;
        while(j<strlen(mot)){
           if(texte[i+j]!=mot[j]){
                non_match=1;
                break;
            }
            j++;
        }
        if(non_match==0)
        {
            if (i==taille){
                return nb_occur;
            }
            nb_occur+=1;     
        }
        non_match=0;
        i++;
    }
}
//---------------------------------------strcmp1---------------------------------------------

int recherche_naive_sans_boucle_rapide_strncmp(char* mot,char* texte)
{
    int nb_occur=0;
    int taille=strlen(texte)-strlen(mot);
    for (int i=0;i<=strlen(texte)-strlen(mot);i++)
    {
        if (strncmp(&texte[i], mot, strlen(mot)) == 0) {
            nb_occur++;
        }
    }
    return nb_occur;
}
//---------------------------------------strcmp2---------------------------------------------
int recherche_naive_avec_boucle_rapide_strcmp(char* mot,char* texte)
{
    int non_match=0;
    int nb_occur=0;
    char firts_lettre=mot[0];
    for (int i=0;i<=strlen(texte)-strlen(mot);i++){

        if (firts_lettre==texte[i]){

            if (strncmp(&texte[i], mot, strlen(mot)) == 0) {
                   nb_occur++;
            }
        }
        
        non_match=0;
    }
    return nb_occur;
}
//---------------------------------------strcmp3---------------------------------------------

int recherche_naive_avec_boucle_rapide_sentinelle_strcmp(char* mot,char* texte){  
            
    texte = (char*)realloc(texte, (strlen(texte) + strlen(mot) + 1) * sizeof(char));
    strcat(texte, mot);
    int non_match=0;
    int nb_occur=0;
    int taille=strlen(texte)-strlen(mot);
    int i=0;
    while(true)
    {
        if (strncmp(&texte[i], mot, strlen(mot)) == 0) {
                   if (i==taille){
                        
                        return nb_occur;
                    }
                    nb_occur++;
            }
            i++;
    }
}


