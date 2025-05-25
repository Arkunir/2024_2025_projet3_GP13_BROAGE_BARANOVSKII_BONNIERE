# GP13_BROAGE_BARANOVSKII_BONNIERE
Projet 3 de NSI, création d'une IA complétant une version abrogée de geometry dash

### Description:
Ce projet est une version originale de Geometry Dash fait de nos propre moyen, un jeu de plateforme classique qui a pour but de sauter au dessus de certain obstacles.  
Le jeu est composé d'un mode infinie avec un système de niveau de vitesse, au bout d'un certain nombre de point récupérer votre vitesse peu augmenter.  
Le jeu est également composé d'une IA qui peut jouer le jeu.  

### Description de l'IA:  

L'IA est basée sur un algorithme de Deep Learning, plus précisément un réseau de neurones.  
L'IA est capable de prendre des décisions en temps réel, elle peut donc jouer d'elle même.   
L'IA est également capable de s'adapter à la difficulté du jeu, elle peut donc réagir en fonction de la difficulté de l'obstacle.  
L'IA est également capable de mémoriser les mouvements qu'elle a déjà effectuer.  

Son code est divisé en 3 parties:  
- Une partie qui gère les mouvements de l'IA et qui apprend de ce qu'il fait.  
- Une partie qui observe et annalyse a 8 bloc les obstacles devant lui.  
- Une partie qui va executé la Q table et qui va choisir la meilleure action a faire.  

### Installation:  
Clonez le dépôt : Ouvrez le liens puis cloner le code grâce à [ce dépôt](https://github.com/Arkunir/2024_2025_projet3_GP13_BROAGE_BARANOVSKII_BONNIERE)

Conditions d'exécution : Assurez-vous d'avoir **Python installé** sur votre machine.

### Utilisation:  
- Installer : Python

- Dans la console faire les commandes suivantes: `pip install pygame` et `pip install numpy`

- Une fois le code lancé choisissez votre mode de jeu 'IA' ou 'Joueur'. Pour sauter au dessus des obstacles vous devez appuyer sur la touche 'Espace'.  

### Status:  
Travail en cour:
- train l'ia

Travail fini:
- Implémenté l'IA
- corps du jeu
- menu

### Contributeur:  



### Licence:  

Ce projet est sous licence libre. Vous êtes libre de l'utiliser, de le modifier et de le distribuer, tant que vous respectez les termes de la licence.
