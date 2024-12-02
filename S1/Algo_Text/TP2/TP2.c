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
//---------------------------------------Morris-Pratt---------------------------------------------
int calcul_bord(char* mot)
{
    int taille=strlen(mot);
    int bord[taille];
    bord[0]=0;
    int i=1,j=0;
    while(i<taille){
        if(mot[i]==mot[j]){
            j++;
            bord[i]=j;
            i++;
        }
        else{
            if(j!=0){
                j=bord[j-1];
            }
            else{
                bord[i]=0;
                i++;
            }
           
        }
    }
    return bord[taille-1];
}
int bon_prefix(char* mot,int indexe){
    if(indexe==0){
        return -1;
    }
    else{
          char *tmp = (char*)malloc(indexe+1 * sizeof(char)); 
          for(int i=0;i<indexe;i++){
            tmp[i]=mot[i];
          }
          return calcul_bord(tmp);
    }
    
}
int Morris_Pratt(char* mot,char* texte){
    int i=0;
    int nb_occurence=0;
    int taille=strlen(texte);
    int bonpref[strlen(mot)];
    for (int j=0;j<strlen(mot);j++)
    { 
        bonpref[j]=bon_prefix(mot,j);
    }
    for (int j=0;j<taille;j++)
    {
        while(i>=0 && mot[i]!=texte[j]){
            i=bonpref[i];
        }
        i++;
        if(i==strlen(mot)){
             nb_occurence++;
             i=bonpref[i];
        }
    }
    return nb_occurence;
}
//---------------------------------------Knuth-Morris-Pratt---------------------------------------------
int meil_prefix(char* mot,int indexe){
    if(indexe==0){
        return -1;
    }
    else{
          char *tmp = (char*)malloc(indexe+1 * sizeof(char)); 
          for(int i=0;i<=indexe;i++){
            tmp[i]=mot[i];
          }
          if(mot[calcul_bord(tmp)]!=mot[indexe]){
            return calcul_bord(tmp);
          }
          else{
             return meil_prefix(mot,calcul_bord(tmp));
          }
    }
    
}
int Knuth_Morris_Pratt(char* mot,char* texte){
    int i=0;
    int nb_occurence=0;
    int taille=strlen(texte);
    int meipref[strlen(mot)];
    for (int j=0;j<strlen(mot);j++)
    { 
        meipref[j]=meil_prefix(mot,j);
    }
    for (int j=0;j<taille;j++)
    {
        while(i>=0 && mot[i]!=texte[j]){
            i=meipref[i];
        }
        i++;
        if(i==strlen(mot)){
             nb_occurence++;
              i=meipref[i];
        }
    }
    return nb_occurence;
}







int main(){
    char *texte = (char*)malloc(20 * sizeof(char));    
    char* mot = (char*)malloc(20 * sizeof(char)); 
    strcpy(texte, "taatatataa");
    strcpy(mot, "tata");
    // int res=recherche_naive_avec_boucle_rapide_sentinelle_strcmp(mot,texte);
    // printf("%d",r0es);
    int res=Knuth_Morris_Pratt(mot,texte);
    printf("Le nbre occurence est %d \n",res);
    
    return 0; 
}