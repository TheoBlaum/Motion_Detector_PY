# Détecteur de Mouvement et de Visages

Ce programme est un système de détection de mouvement et de visages en temps réel utilisant la webcam. Il combine la détection de mouvement avec la reconnaissance faciale pour fournir une surveillance visuelle complète.

## Fonctionnalités

- Détection de mouvement en temps réel
- Détection de visages
- Affichage des FPS (images par seconde)
- Horodatage des images
- Interface visuelle avec informations en temps réel

## Prérequis

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Assurez-vous d'avoir Python installé sur votre système
2. Installez les dépendances nécessaires :
```bash
pip install opencv-python numpy
```

## Utilisation

1. Exécutez le programme :
```bash
python motion_detector.py
```

2. Contrôles :
- Appuyez sur 'q' pour quitter le programme

## Affichage

Le programme affiche :
- Le nombre de visages détectés
- Les FPS en temps réel
- La date et l'heure actuelles
- Des rectangles bleus autour des visages détectés
- Des rectangles verts autour des zones de mouvement

## Configuration

Les paramètres suivants peuvent être ajustés dans le code :
- `motion_threshold` : Seuil de détection de mouvement
- `face_scale_factor` : Facteur d'échelle pour la détection de visages
- `face_min_neighbors` : Nombre minimum de voisins pour la détection de visages
- `face_min_size` : Taille minimale des visages
- `blur_kernel` : Taille du noyau de flou

## Structure du Code

Le programme utilise une architecture multi-thread pour :
- Capture vidéo en continu
- Traitement des images
- Détection de mouvement et de visages
- Affichage des résultats

## Notes

- Le programme essaie automatiquement différentes sources de caméra (0, 1, -1)
- Les performances peuvent varier selon la puissance de votre ordinateur
- La détection de visages utilise le classifieur Haar Cascade d'OpenCV 