# GP13_BROAGE_BARANOVSKII_BONNIERE
Projet 3 de NSI, création d'une IA complétant une version abrogée de Geometry Dash

### Description:
Ce projet est une version originale de Geometry Dash faite de nos propres moyens, un jeu de plateforme classique qui a pour but de sauter au-dessus de certains obstacles.  
Le jeu est composé d'un mode infini avec un système de niveau de vitesse, au bout d'un certain nombre de points récupérés votre vitesse peut augmenter.  
Le jeu est également composé d'une IA qui peut jouer au jeu.  

### Description de l'IA:  

L'IA est basée sur un algorithme de Deep Learning, plus précisément un réseau de neurones.  
L'IA est capable de prendre des décisions en temps réel, elle peut donc jouer d'elle-même.   
L'IA est également capable de s'adapter à la difficulté du jeu, elle peut donc réagir en fonction de la difficulté de l'obstacle.  
L'IA est également capable de mémoriser les mouvements qu'elle a déjà effectués.  

Son code est divisé en 3 parties:  
- Une partie qui gère les mouvements de l'IA et qui apprend de ce qu'elle fait.  
- Une partie qui observe et analyse à 8 blocs les obstacles devant lui.  
- Une partie qui va exécuter la Q table et qui va choisir la meilleure action à faire.  

### Fonctionnalités détaillées

#### Gameplay principal
- Le joueur contrôle un cube qui peut sauter pour éviter des obstacles dans un jeu de plateforme à défilement horizontal.
- Le saut est contrôlé par la touche 'Espace'.
- Le score augmente à mesure que le joueur évite les obstacles.
- La vitesse de défilement du jeu augmente progressivement selon des seuils de score, rendant le jeu plus difficile.
- Le joueur peut changer de skin aléatoirement à chaque changement de vitesse, avec des skins chargés depuis le dossier `skins/`.

#### Types d'obstacles
- **Pics simples et multiples** : obstacles en forme de pics simples, doubles, triples ou quadruples à éviter.
- **Blocs** : plateformes sur lesquelles le joueur peut se tenir.
- **Blocs avec pics** : combinaisons de blocs avec des pics pour augmenter la difficulté.
- **Obstacles rebondissants** : obstacles qui rebondissent verticalement, ajoutant un défi supplémentaire.
- **Pilier de blocs** : structures composées de plusieurs blocs empilés.
- **Jump Pads** : plateformes qui propulsent le joueur vers le haut avec une force de saut accrue.
- **Orbes jaunes et violettes** : objets spéciaux qui, lorsqu'activés, donnent un boost de saut au joueur.
- **Obstacles combinés** : combinaisons complexes d'obstacles, pics et jump pads.

#### IA par apprentissage par renforcement
- Implémentation d'une IA utilisant Q-learning pour apprendre à jouer au jeu.
- L'IA observe l'état du jeu (distance aux obstacles, type d'obstacle, vitesse, position et vitesse du joueur, etc.) et choisit des actions (sauter ou attendre).
- L'IA est capable de mémoriser ses expériences et d'améliorer ses performances au fil des épisodes d'entraînement.
- Les modèles d'IA sont sauvegardés et peuvent être rechargés pour continuer l'entraînement ou jouer avec la meilleure IA.

#### Mode Meilleure IA
- Permet de jouer avec la meilleure IA entraînée, sans exploration (epsilon=0).
- L'IA utilise uniquement ses connaissances acquises pour jouer de manière optimale.

#### Visualisation des données d'entraînement
- Un écran affiche un graphique des scores obtenus par l'IA au fil des épisodes d'entraînement.
- Le graphique montre les scores bruts, la moyenne mobile, et des statistiques comme le score maximum, minimum, et la progression.

#### Menu principal
- Interface avec boutons pour choisir entre :
  - Mode Joueur (contrôle manuel)
  - Mode IA par renforcement (entraînement de l'IA)
  - Mode Meilleure IA (jouer avec la meilleure IA)
  - Visualisation du graphique d'entraînement

#### Utilisation des assets
- Les skins du joueur sont chargés depuis le dossier `skins/` contenant des images PNG ou JPG.
- Si aucun skin n'est trouvé, un skin par défaut (un carré bleu) est utilisé.
- Les skins sont redimensionnés et appliqués au joueur, avec rotation lors des sauts.

### Installation:  
Clonez le dépôt : Ouvrez le lien puis clonez le code grâce à [ce dépôt](https://github.com/Arkunir/2024_2025_projet3_GP13_BROAGE_BARANOVSKII_BONNIERE)

Conditions d'exécution : Assurez-vous d'avoir **Python installé** sur votre machine.

### Utilisation:  
- Installer : Python

- Dans la console, faites les commandes suivantes : `pip install pygame` et `pip install numpy`

- Une fois le code lancé, choisissez votre mode de jeu 'IA' ou 'Joueur'. Pour sauter au-dessus des obstacles, vous devez appuyer sur la touche 'Espace'.  

### Status:  
Travail en cours:

Travail achevé:
- Implémentation de l'IA
- corps du jeu
- menu
- graphiques
- mode de test de l'IA sans mutation
- mode de jeu pour joueur

### Contributeurs:  

BARANOVSKI Roman, BONNIERE Tristan, BROAGE Theodore 1ère4

### Licence:  

Ce projet est sous licence libre. Vous êtes libre de l'utiliser, de le modifier et de le distribuer, tant que vous respectez les termes de la licence.
