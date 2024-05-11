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

TODO: 1, 2, 3 for part 2

TODO: 1, 2, 3 for part 3

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

Afin de pouvoir d'augmenter la complexité du modèle pour qu'il apprenne encore plus, tout en évittant l'overfitting nous avons augmenté les dropout à 2 fois 0.5, tout l'enjeu est d'arriver à donner au modèle un moyen de ne pas pouvoir se concentrer sur des mauvaises caractéristiques au lieu enlevant une partie des poids synaptiques à chaque batch, mais tout en lui laissant assez d'informations pour qu'il puisse quand même arriver à bien généraliser. En effet, si augmente trop le Dropout la performance recommence à redescendre...

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
```python
l0 = Input(shape=(height, width, 1), name='l0')
l1 = Conv2D(50, (10,10), padding='same', activation='relu', name='l1')(l0)
l1_mp = MaxPooling2D(pool_size=(2,2), name='l1_mp')(l1)
drop1 = Dropout(0.5)(l1_mp)
l2 = Conv2D(50, (10,10), padding='same', activation='relu', name='l2')(drop1)
l2_mp = MaxPooling2D(pool_size=(2,2), name='l2_mp')(l2)
drop2 = Dropout(0.5)(l2_mp)
l3 = Conv2D(50, (10,10), padding='same', activation='relu', name='l3')(drop2)
l3_mp = MaxPooling2D(pool_size=(2,2), name='l3_mp')(l3)
flat = Flatten(name='flat')(l3_mp)
l4 = Dense(20, activation='relu', name='l4')(flat)
l5 = Dense(n_classes, activation='softmax', name='l5')(l4)
model = Model(inputs=l0, outputs=l5)
model.summary()
```

**Nombre de poids synaptiques**

Calculs:
1. Couche de convolution: nombres de filtres * largeur filtre * hauteur filtre + autant de biais que de nombre de filtres <!-- todo: check ce calcul--> . Ici pour 50 filtres de 10x10 on aura `50*10*10 + 50= 5050` (pour L1, L2, L3)
1. Couche de maxpooling et dropout: pas de poids synaptiques, c'est juste une transformation intermédiaire
1. Couche de perceptrons: nombre de perceptrons * nombre de valeurs d'entrée + autant de biais que de perceptrons. Ici pour L4: `20 * 450 (après flatten) + 20 biais = 9020`. Pour L5: `10 * 20 (20 sorties car 20 neurones couches précédentes) + 10 biais = 210`

TOTAL: `3 * 5050 + 920 + 210 = 16280` TODO: résultat et calculs pas du tout sûr, difficile à trouver comment bien les calculer.


<!-- The CNNs models are deeper (have more layers), do they have more weights than the
shallow ones? explain with one example. -->

<!-- TODO: pas sur de comprendre comment y répondre à ça, ya une slide chap 7 qui parle de ça... -->


<!-- 4. Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete the
code to reproduce the architecture plotted in the notebook. Present the confusion matrix,
accuracy and F1-score of the validation and test datasets and discuss your results. -->