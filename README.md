# Labo 3 - ARN
Auteurs: Felix Breval et Samuel Roland

## Introduction
TODO

<!-- What is the learning algorithm being used to optimize the weights of the neural networks? -->
<!-- What are the parameters (arguments) being used by that algorithm? -->
<!-- What loss function is being used ? -->
<!-- Please, give the equation(s) -->

<!-- For each experiment excepted the last one (shallow network learning from raw data, -->
<!-- shallow network learning from features and CNN): -->

<!-- 1. Select a neural network topology and describe the inputs, indicate how many are
they, and how many outputs? -->

<!-- 2. Compute the number of weights of each model (e.g., how many weights between the
input and the hidden layer, how many weights between each pair of layers, biases,
etc..) and explain how do you get to the total number of weights. -->

<!-- 3. Test at least three different meaningful cases (e.g., for the MLP exploiting raw data,
test different models varying the number of hidden neurons, for the feature-based
model, test pix_p_cell 4 and 7, and number of orientations or number of hidden
neurons, for the CNN, try different number of neurons in the feed-forward part)
describe the model and present the performance of the system (e.g., plot of the
evolution of the error, nal evaluation scores and confusion matrices). Comment the
differences in results. Are there particular digits that are frequently confused? -->

### Partie 2 - Digits recognition sur données brutes

Pour cette partie il s'agit de juste appliquer ce que nous avons pu apprendre à faire dans le laboratoire 2 et le faire pour cette base de donnée.

**Sélection de la topologie du réseau neuronal :**
Le modèle de réseau neuronal utilisé dans le code est une architecture de type MLP (Multi-Layer Perceptron) avec une couche cachée et une couche de sortie. Les entrées sont des vecteurs plats représentant des images de 28x28 pixels (784 dimensions), et il y a 10 classes de sortie correspondant aux chiffres de 0 à 9 dans le jeu de données MNIST.

**Calcul du nombre de poids pour chaque modèle :**

Entre la couche d'entrée et la couche cachée : 784 (entrées) * 150 (neurones cachés) + 150 (biais) = 117,750 poids.

Entre la couche cachée et la couche de sortie : 150 (neurones cachés) * 10 (sorties) + 10 (biais) = 1,510 poids.

Total des poids : 117,750 + 1,510 = 119,260 poids.

**Différentes approches**:

Dans cette partie, le paramètre qui nous intéresse est principalement le nombre de neurones cachés. sachant que nous avons 784 entrées (28x28), on sait qu'au maximum on voudra 392 (784/2) neurones cachés.

Nous avons testé pour 392, 300, 200, 100 puis avons remarqué que 150 était une valeur qui semblait être la meilleure.

Il y avait un clair overfitting pour 392 neurones, de même pour 300 et 200.

Pour 100 on avait un underfitting trop élevé.

Un dropout a aussi été ajouté, de hauteur de 50% afin de finir avec un accuracy de 98%.

Le nombre d'epochs a aussi été augmenté à 35.



**Résultats**:

Nous finissons avec un f1-score de 0.9798

### Partie 3 - Digits recognition depuis des features dans les inputs

**Sélection de la topologie du réseau neuronal** :
Le modèle utilisé dans ce code est un MLP (Multi-Layer Perceptron) qui utilise les Histogrammes des Gradients Orientés (HOG) comme caractéristiques d'entrée au lieu des données brutes. Chaque image est transformée en un vecteur HOG, et ces vecteurs sont utilisés comme entrée pour le réseau.

**Calcul du nombre de poids pour chaque modèle** :

Entre la couche d'entrée et la couche cachée : Le nombre de poids est 200*392+200 (pour une couche cachée de 200 neurones et un vecteur HOG de taille 392) ce qui donne 78500 poids.

Entre la couche cachée et la couche de sortie : Le nombre de poids est 200*10+10 (pour 200 neurones dans la couche cachée et 10 classes de sortie) ce qui donne 2010 poids.

Total des poids : 80510 poids.

**Différentes approches**:

Pour ce modèle basé sur les caractéristiques, nous avons réalisé plusieurs tests significatifs. Tout d'abord, nous avons testé différentes valeurs pour les hyperparamètres tels que le nombre de pixels par cellule (pix_p_cell) en essayant les valeurs 4 et 7, tout en conservant le nombre d'orientations à 8 car cela semblait plus cohérent. Nous avons observé comment ces changements affectent les performances du modèle.
De plus, nous avons varié le nombre de neurones dans la couche cachée en testant les valeurs 100, 200 et 300. Après une évaluation des performances, nous avons décidé de finalement utiliser 200 neurones dans la couche cachée, car cela fournissait un bon compromis entre la capacité de représentation et la complexité du modèle.

Après avoir entraîné chaque modèle avec les différentes configurations, nous avons évalué leurs performances en traçant l'évolution de l'erreur, en présentant les scores d'évaluation finaux et les matrices de confusion. Nous avons fini par prendre un pix_p_cell de 7 (avec une orientation de 8 et 200 neurones). Cela nous a donné un bon résultat qui, comparé aux autres tests, n'overfit pas.

## Partie 4 - CNN sur digits
Après beaucoup de tests différents pour tenter de mieux comprendre l'impact de chaque type d'hyperparamètre, voici les essais pertinents et réflexions autour que nous avons pu faire. La plupart des tests ont tournés sur 15 ou 20 epochs, le but était d'avoir un nombre petit pour pouvoir faire plein de tests différents (entre 10s et 1 minute d'exécution avec une carte graphique correcte). En testant plus d'épochs sur une plus longue durée, on se rend compte que l'amélioration est possible mais bien faible et l'overfitting devient beaucoup plus compliqué à éviter (sans compter qu'il devient très lent de tester des changements de paramètres). Nous nous sommes rendus compte que les performances sont plus variables qu'un MLP, en relançant plusieurs fois la même configuration, il semble il y avoir beaucoup plus d'impact de l'aléatoire sur le résultat, changeant parfois 1-3% de différence d'accuracy finale, cela compliquait l'analyse de micro améliorations, était-ce de la chance ou cela allait-il vraiment dans la bonne direction ?

Au tout début, la configuration fournie donnait des résultats très très bizarre, une fois 31% d'accuracy, une autre fois 50%, puis 65%, en bref des valeurs qui changaient du tout au tout à chaque exécution. Cela était du au 2 neurones de la couche cachée du MLP de sortie, après recommendation du prof nous sommes partis sur 20 neurones, cette valeur a été changé un peu plus haut et plus bas mais cette valeur a l'air d'être effectivement la plus optimale. A ce moment, l'accuracy atteinte a été très vite sur ~60% et restait dans ces alentours.

![](imgs/2024-05-11_04-41.png)

Comme demandé, nous avons tenté de changer la taille des filtres, le nombre de filtres par couches, et d'ajouter plus ou moins de dropout. En poussant à beaucoup plus grand la taille des filtres et le nombre de filtres par couche, nous avons été étonnés de voir très vite monté l'accuracy de 88, puis 91, puis vers 96, 97, 98 et même 99% après beaucoup de tentatives. La seule chose que nous n'avons pas vraiment tenté de changer est la `pool_size=(2,2)` de `MaxPooling2D` comme cela n'était pas demandé et cela nous donnait des erreurs dû à des tailles incompatibles.

*Un exemple de résultat à 99%, le but étant de monté le dizième de pourcent au plus haut tout en gardant l'overfitting minimal. Ici, on voit qu'il y en a un peu.*

![](imgs/2024-05-10_19-23_10.png)

IL y avait également très souvent des configurations qui nous donnait une courbe d'entrainement significativement plus haute que celle d'entrainement, nous indiquant underfitting à cause que notre modèle n'apprend pas assez/qu'il est trop simple.

![](imgs/loin.png)


Pour faciliter la vie, nous avons gardé les mêmes configurations pour les 3 couches dans les essais suivants:

Nous avons d'abord bien poussé le nombre de filtres par couches, jusqu'à 50

![](imgs/2024-05-10_19-23_8.png)

Nous avons continué de monter (100 filtres) pour continuer sur ce gros gain de performance (90% -> 99%), mais l'overfitting nous rattrape...

![](imgs/overfitting.png)

Nous avons ensuite tenté de monter la taille des filtres, cela n'améliore que légèrement ici...

![](imgs/2024-05-10_19-23_7.png)
![](imgs/2024-05-10_18-55_1.png)

Ce qui est intéressant c'est de voir que dès qu'on rebaisse de 50 à 40 de nombre de filtres, la performance est un poil moins bonne et on voit tout de suite que la distance entre les 2 courbes se creusent à nouveau. Ce paramètre a l'air crucial pour que le modèle apprenne assez.

![](imgs/2024-05-10_18-46.png)

Même en augmentant la taille des filtres, on n'arrive pas à rejoindre les performances précédentes si le nombre de filtres est réduit.
![](imgs/2024-05-10_18-55.png)

![](imgs/2024-05-10_18-58.png)
![](imgs/2024-05-10_18-55_2.png)

![](imgs/2024-05-10_19-01.png)

Nous avons ensuite ajouté du dropout simple de 0.1 après la première couche de Conv2D + MaxPooling2D.

![](imgs/2024-05-10_19-24_2.png)

Puis avec 2 dropout intermédiaires

![](imgs/2024-05-10_19-24.png)

![](imgs/2024-05-10_19-24_1.png)

Peut-être que d'avoir un deuxième dropout plus petit est plus fin en terme d'impact.

Afin de pouvoir d'augmenter la complexité du modèle pour qu'il apprenne encore plus, tout en évitant l'overfitting nous avons augmenté les dropout à 2 fois 0.5, tout l'enjeu est d'arriver à donner au modèle un moyen de ne pas pouvoir se concentrer sur des mauvaises caractéristiques au lieu enlevant une partie des poids synaptiques à chaque batch, mais tout en lui laissant assez d'informations pour qu'il puisse quand même arriver à bien généraliser. En effet, si augmente trop le Dropout la performance recommence à redescendre...

Nous avons aussi testé des configurations plus mélangées juste pour l'avoir testé une fois, cela ne donne rien de très facile à analyser ni de meilleur que précédemment...

![](imgs/2024-05-10_19-23.png)

Voici la confirmation de notre intuition que laisser beaucoup plus d'epochs ne servira a rien, l'overfitting est énorme ici.

![](imgs/2024-05-10_22-07.png)
![](imgs/2024-05-10_22-13.png)

A un moment donné, la configuration suivante (3 fois 30 filtres de 10x10, max pooling de 2x2 toujours, avec 2 fois 0.5 de dropout intermédiaire) a donné une accuracy de **99.5%** ! C'était plus un coup de chance car la même configuration n'a jamais redonné ce même record pour ce labo.

![](imgs/2024-05-10_17-43.png)


---

Notre modèle finale lancé sur 50 epochs performe avec une accuracy de **99.4%**, avec un léger overfitting dès 20 epochs.
![modelefinale.png](imgs/modelefinale.png)

La matrice de confusion de ce modèle est la suivante
```
array([[ 979,    0,    0,    0,    0,    0,    0,    1,    0,    0],
       [   0, 1133,    1,    1,    0,    0,    0,    0,    0,    0],
       [   2,    1, 1024,    0,    0,    0,    1,    3,    1,    0],
       [   0,    0,    0, 1006,    0,    3,    0,    0,    1,    0],
       [   0,    0,    0,    0,  976,    0,    2,    0,    0,    4],
       [   1,    0,    0,    1,    0,  889,    1,    0,    0,    0],
       [   2,    2,    0,    0,    0,    1,  953,    0,    0,    0],
       [   0,    2,    3,    1,    1,    0,    0, 1020,    0,    1],
       [   2,    1,    1,    0,    0,    2,    0,    0,  967,    1],
       [   0,    0,    0,    0,    5,    3,    0,    0,    0, 1001]])
```

On y voit qu'il y a très peu d'erreurs, excepté quelques petites confusions plus marquées que d'autres: 3 fois le 2 classifié en 7, 3 fois le 3 en 5, 4 fois le 4 den 9, 3 fois le 7 en 2 (sens inverse), 5 fois le 9 classifié en 4 et 3 fois le 9 en 5. Ces confusions ont une certaine logique, les chiffres confondues sont généralement assez proches en termes de traits qui les composent. Par ex: 2 confondu avec 7, il y a 2 branches communes dans leur écriture.

**Topologie**

La topologie de notre réseau est la suivante:
1. **Entrée** images de 28x28 sur 1 canal (noir/blanc)
1. L1: Couche de **convolution**: **50 filtres de 10x10** avec padding et fonction d'activation `relu`
1. L1_MP: Couche de **maxpooling**: taille de filtre 2x2
1. **Dropout de 0.5**
1. L2: Couche de **convolution**: **50 filtres de 10x10** avec padding et fonction d'activation `relu`
1. L2_MP: Couche de **maxpooling**: taille de filtre 2x2
1. **Dropout de 0.5**
1. L3: Couche de **convolution**: **50 filtres de 10x10** avec padding et fonction d'activation `relu`
1. L3_MP: Couche de **maxpooling**: taille de filtre 2x2
1. Flatten des images
1. L4: Couche cachée de **perceptrons**: **20** neurones, fonction d'activation `relu`
1. L5: Couche de sortie de **perceptrons**: 10 neurones car 10 classes (chiffres de 0 à 9), fonction d'activation `softmax`

Nous n'avons pas changé la loss function (toujours `categorical_crossentropy`) et l'optimizer `RMSprop`.

**Nombre de poids synaptiques**

Calculs:
1. Couche de convolution: nombres de filtres * largeur filtre * hauteur filtre + autant de biais que de nombre de filtres <!-- todo: check ce calcul--> . Ici pour 50 filtres de 10x10 on aura `50*10*10 + 50= 5050` (pour L1, L2, L3)
1. Couche de maxpooling et dropout: pas de poids synaptiques, c'est juste une transformation intermédiaire
1. Couche de perceptrons: nombre de perceptrons * nombre de valeurs d'entrée + autant de biais que de perceptrons. Ici pour L4: `20 * 450 (après flatten) + 20 biais = 9020`. Pour L5: `10 * 20 (20 sorties car 20 neurones couches précédentes) + 10 biais = 210`

TOTAL: `3 * 5050 + 920 + 210 = 16280` TODO: résultat et calculs pas du tout sûr, difficile à trouver comment bien les calculer.

## Remaining Questions

**The CNNs models are deeper (have more layers), do they have more weights than the shallow ones? explain with one example**:

En général, les réseaux de neurones convolutifs (CNN) peuvent avoir plus de poids que les réseaux peu profonds (avec moins de couches), mais cela dépend de plusieurs facteurs, y compris la taille des filtres, le nombre de filtres par couche, et la taille des couches cachées dans les réseaux peu profonds.

Prenons un exemple simple pour illustrer cela :

Considérons deux modèles :
1. **Modèle Peu Profond (Shallow)** :
   - Ce modèle a une seule couche cachée entièrement connectée avec 200 neurones.
   - Supposons que la taille du vecteur d'entrée soit de 392 (comme dans notre exemple de MLP basé sur les caractéristiques HOG).
   - Calcul des poids entre la couche d'entrée et la couche cachée : \( 200 \times 392 + 200 \) (pour les poids) + \( 200 \) (biais) = 78500 poids.
   - Ce modèle a un total de 78500 poids.

2. **Modèle CNN Profond (Deep)** :
   - Ce modèle a plusieurs couches de convolution suivies de couches de pooling, puis une ou plusieurs couches entièrement connectées.
   - Supposons que ce modèle ait trois couches de convolution avec 32 filtres chacune, suivies de couches de pooling, puis une couche entièrement connectée avec 200 neurones.
   - Les poids dans les couches de convolution sont partagés, ce qui réduit le nombre de paramètres par rapport aux couches entièrement connectées.
   - Calculons les poids uniquement pour les couches entièrement connectées :
     - Entre la dernière couche de pooling et la couche entièrement connectée : \( 7 \times 7 \times 32 \times 200 + 200 \) (pour les poids) + \( 200 \) (biais) ≈ 313,000 poids.
   - Supposons que le nombre total de poids dans les couches de convolution soit d'environ 10,000 (ce nombre peut varier en fonction de la taille des filtres et du nombre de filtres).
   - Ce modèle pourrait avoir un total d'environ 323,000 poids.

Dans cet exemple, bien que le modèle CNN ait plus de couches, la plupart des poids sont concentrés dans les couches entièrement connectées, en particulier dans la dernière couche avant la sortie. Cela peut conduire à un nombre total de poids plus élevé dans le modèle CNN profond par rapport au modèle peu profond.

**Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete the code to reproduce the architecture plotted in the notebook. Present the confusion matrix, accuracy and F1-score of the validation and test datasets and discuss your results.**:

### Ensemble de validation :
- **Matrice de confusion** :
  ```
  [[6 2]
   [2 6]]
  ```
  - Vrai négatif (TN) : 6
  - Faux positif (FP) : 2
  - Faux négatif (FN) : 2
  - Vrai positif (TP) : 6

- **Précision** : 93%
- **Score F1** : 0,75

La matrice de confusion montre que sur 16 échantillons de validation, 12 ont été correctement classés (6 vrais négatifs et 6 vrais positifs), et 4 ont été mal classés (2 faux positifs et 2 faux négatifs). La précision de 93% indique une proportion élevée d'échantillons correctement classés. Le score F1 de 0,75 suggère un équilibre entre la précision et le rappel.

### Ensemble de test :
- **Matrice de confusion** :
  ```
  [[121 113]
   [  6 384]]
  ```
  - Vrai négatif (TN) : 121
  - Faux positif (FP) : 113
  - Faux négatif (FN) : 6
  - Vrai positif (TP) : 384

- **Précision** : Non fournie
- **Score F1** : 0,8658399098083427

La matrice de confusion montre que sur 624 échantillons de test, 505 ont été correctement classés (121 vrais négatifs et 384 vrais positifs), et 119 ont été mal classés (113 faux positifs et 6 faux négatifs). Le score F1 de 0,87 indique un bon équilibre entre la précision et le rappel sur l'ensemble de test.

### Discussion :
- Les ensembles de validation et de test montrent une bonne performance avec une précision et un score F1 élevés.
- Cependant, on observe une augmentation notable des faux positifs dans l'ensemble de test par rapport à l'ensemble de validation, ce qui pourrait indiquer un surajustement potentiel ou des problèmes de généralisation.
- Une analyse plus approfondie est nécessaire pour comprendre pourquoi le modèle se comporte différemment sur l'ensemble de test par rapport à l'ensemble de validation. Cela pourrait impliquer d'examiner la nature des échantillons mal classés et les biais potentiels dans les données ou le modèle.
