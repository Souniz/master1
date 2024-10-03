struct Trie
{
    int maxNode; /* Nombre maximal de noeuds du trie */
    int nextNode; /* Indice du prochain noeud disponible */
    int **transition; /* matrice de transition */
    char *finite; /* etats terminaux */
};
typedef struct Trie Trie;
struct _list {
    int startNode; /* etat de depart de la transition */
    int targetNode; /* cible de la transition */
    unsigned char letter; /* etiquette de la transition */
    struct _list *next; /* maillon suivant */
};
typedef struct _list *List;



Trie createTrie(int maxNode);
void insertInTrie(Trie trie, unsigned char *w);
int isInTrie(Trie trie, unsigned char *w);