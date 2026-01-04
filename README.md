# 📚 FICHE DE RÉVISION - TP4 POINTNET

## 🎯 PARTIE 1 : BUT ET CONCEPT DU TP

### Objectif Principal

Le but de ce TP est de comprendre **PointNet**, un réseau de neurones qui travaille directement sur des **nuages de points 3D** pour les classifier.

### Problème à Résoudre

**Input** : Un nuage de points 3D (ex: 2048 points avec coordonnées (x, y, z))  
**Output** : La classe de l'objet (cylindre, rectangle, tore)

### Challenges Uniques des Nuages de Points

**1. Ordre des points variable (Permutation Invariance)**

```python
[p1, p2, p3] = [p3, p1, p2] = [p2, p3, p1]
```

Le réseau doit donner **la même prédiction** peu importe l'ordre des points.

**2. Rotation aléatoire**

L'objet peut être tourné dans n'importe quelle direction → besoin d'invariance aux rotations.

**3. Pas de structure régulière**

- **Images** : Grille régulière de pixels avec voisinage spatial
- **Nuages de points** : Points désordonnés sans structure fixe

### Solution : PointNet

**3 idées clés** :

1. **Conv1d (kernel=1)** : Traiter chaque point indépendamment
2. **MaxPooling global** : Obtenir une feature invariante à l'ordre
3. **T-Net** : Apprendre à aligner les objets automatiquement

**Analogie** : C'est comme reconnaître un objet en regardant une poignée de points dans l'espace 3D, peu importe comment ils sont ordonnés ou orientés.

---

## 🎲 PARTIE 2 : GÉNÉRATION DES DONNÉES

### Code de Génération

```python
def sample_cylinder(npts=2048):
    # Génère un cylindre avec 2048 points sur la surface
    
def sample_rectangle(npts=2048):
    # Génère un parallélépipède
    
def sample_torus(npts=2048):
    # Génère un tore
    
def apply_random_rotation(x):
    # Rotation aléatoire 3D
    
def normalize(x):
    # Centrage + normalisation [-1, 1]
```

### Échantillonnage du Cylindre - Explication Détaillée

#### Calcul des Aires

```python
# Paramètres aléatoires
l = np.random.rand()  # Longueur [0, 1]
r = np.random.rand()  # Rayon [0, 1]

# Calcul des aires
a1 = 2 * π * r²        # Aire des 2 cercles (caps haut et bas)
a2 = 2 * π * r * l     # Aire du tube (surface latérale)
```

#### Stratégie de Répartition

**Principe** : Répartir les points **proportionnellement aux aires** pour un échantillonnage uniforme.

**Exemple concret** :
- Si a1 = 0.3 (30% de l'aire totale) et a2 = 0.7 (70%)
- Alors 30% des points sur les cercles, 70% sur le tube

```python
nptscirc = int(npts * a1 / (a1 + a2))  # Points sur les cercles
nptstube = npts - nptscirc              # Points sur le tube
```

#### Échantillonnage Uniforme sur un Disque

```python
u = np.random.rand(nptscirc, 2)
u[:,0] = np.sqrt(u[:,0])  # ⚠️ Transformation cruciale !

# Coordonnées polaires → cartésiennes
x = u[:,0] * r * np.cos(2π * u[:,1])
y = u[:,0] * r * np.sin(2π * u[:,1])
```

#### Pourquoi `sqrt(u)` ? ⚠️ QUESTION D'EXAMEN

**Sans sqrt** : Les points se concentrent au centre (densité non uniforme)  
**Avec sqrt** : Distribution uniforme sur le disque

**Raison mathématique** : L'aire d'un anneau augmente avec le rayon.

**Visualisation** :

```
Sans sqrt:          Avec sqrt:
  ●●●●●              ● ● ●
  ●●●●●             ● ● ● ●
  ●●●●●              ● ● ●
(densité au centre) (uniforme)
```

**Explication** : En coordonnées polaires, l'élément d'aire est $dA = r \cdot dr \cdot d\theta$. Pour compenser le facteur $r$, on échantillonne $r$ selon $\sqrt{u}$ au lieu de $u$.

### Rotation Aléatoire

```python
def apply_random_rotation(x, dim=3):
    # Génère une base orthonormée aléatoire
    u = random_vector()
    v = random_vector() - projection(u onto v)
    w = cross(u, v)
    
    M = [u, v, w]  # Matrice orthogonale
    x = x @ M      # Rotation
    return normalize(x)
```

**Propriété mathématique** : $M$ est une **matrice de rotation** → $M^T M = I$ et $\det(M) = 1$

**Effet** : Chaque objet est tourné aléatoirement dans l'espace 3D.

---

## 🔧 PARTIE 3 : T-NET (Transformation Network)

### Code du T-Net

```python
class MyTNet(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim
        
        # ENCODEUR : Extraire features globales
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        
        # DÉCODEUR : Prédire la matrice de transformation
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim*dim)
    
    def forward(self, x):
        # x: (B, dim, N) - B batches, dim canaux, N points
        
        x = F.relu(self.conv1(x))    # (B, 64, N)
        x = F.relu(self.conv2(x))    # (B, 128, N)
        x = F.relu(self.conv3(x))    # (B, 1024, N)
        
        # MaxPooling global → feature globale
        x = torch.max(x, dim=2)[0]   # (B, 1024)
        
        x = F.relu(self.linear1(x))  # (B, 512)
        x = F.relu(self.linear2(x))  # (B, 256)
        x = self.linear3(x)          # (B, dim*dim)
        
        # Ajouter la matrice identité
        I = torch.eye(self.dim).view(1, dim*dim).repeat(x.size(0), 1)
        x = x + I
        
        # Reshape en matrice
        x = x.view(-1, self.dim, self.dim)  # (B, dim, dim)
        return x
```

### Concept du T-Net

**Objectif** : Apprendre une **matrice de transformation** pour aligner automatiquement l'objet dans une orientation canonique.

**Analogie** : Comme un photographe qui ajuste l'angle de la caméra pour toujours voir l'objet de face, peu importe sa position initiale.

### Exemple Concret avec 3 Points

#### Input : Cylindre Tourné

```python
x = [
  [0.5, 0.8, 0.2],  
  [0.3, 0.6, 0.9],  
  [0.7, 0.4, 0.1]   
]
# Shape: (1, 3, 3) = (batch=1, canaux_xyz=3, points=3)
```

#### ÉTAPE 1 : Conv1d avec kernel=1

```python
self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
```

**Effet** : Applique une transformation linéaire **indépendamment** à chaque point.

**Mathématiquement** : Pour chaque point $p_i$, on calcule $h_i = W \cdot p_i + b$

```python
# Pour chaque point, on applique le même poids W (64×3)
# Point 1: [0.5, 0.8, 0.2] → [f1, f2, ..., f64]  (64 features)
# Point 2: [0.3, 0.6, 0.9] → [g1, g2, ..., g64]  (64 features)
# Point 3: [0.7, 0.4, 0.1] → [h1, h2, ..., h64]  (64 features)

# Output: (1, 64, 3)
```

**Pourquoi kernel=1 ?** ⚠️ QUESTION D'EXAMEN

- On traite chaque point **séparément**
- Pas de notion de voisinage spatial (contrairement aux images)
- Les points n'ont pas d'ordre fixe → pas de convolution classique

**Différence avec Conv2d** :

| Aspect | Conv2d (Images) | Conv1d kernel=1 (Points) |
|--------|----------------|--------------------------|
| **Voisinage** | Kernel 3×3, 5×5 | Pas de voisinage |
| **Structure** | Grille régulière | Points désordonnés |
| **Opération** | Agrège pixels voisins | Transforme chaque point |

#### ÉTAPE 2 : MaxPooling Global

```python
x = torch.max(x, dim=2)[0]  # (B, 1024, N) → (B, 1024)
```

**Effet** : Pour chaque feature (1024 au total), garder la **valeur maximale** parmi tous les points.

**Exemple concret** :

```python
# Après Conv3: (1, 1024, 3)
x = [
  [  # Feature 0
    [0.5, 0.8, 0.2]  → max = 0.8
  ],
  [  # Feature 1
    [0.3, 0.9, 0.6]  → max = 0.9
  ],
  [  # Feature 2
    [0.7, 0.4, 0.1]  → max = 0.7
  ],
  ...
  [  # Feature 1023
    [0.2, 0.5, 0.3]  → max = 0.5
  ]
]

# Après MaxPool: (1, 1024)
x = [0.8, 0.9, 0.7, ..., 0.5]
```

**Propriété CRUCIALE** : Le MaxPool est **invariant à l'ordre** des points ! ⚠️

$$\max([0.5, 0.8, 0.2]) = \max([0.2, 0.5, 0.8]) = \max([0.8, 0.2, 0.5]) = 0.8$$

**Pourquoi cette propriété est essentielle** : Peu importe l'ordre des points en entrée, la feature globale reste la même → le réseau est **permutation invariant**.

#### ÉTAPE 3 : Prédiction de la Matrice

```python
x = self.linear3(x)  # (B, 9) pour dim=3

# Exemple de sortie (petits résidus)
x = [0.1, 0.05, -0.02, 0.03, 0.08, 0.01, -0.01, 0.02, 0.09]

# Ajout de l'identité
I = [1, 0, 0, 0, 1, 0, 0, 0, 1]
x = x + I = [1.1, 0.05, -0.02, 0.03, 1.08, 0.01, -0.01, 0.02, 1.09]

# Reshape en matrice 3×3
M = [[1.1,  0.05, -0.02],
     [0.03, 1.08,  0.01],
     [-0.01, 0.02, 1.09]]
```

**Pourquoi ajouter l'identité ?** ⚠️ QUESTION D'EXAMEN

**Sans identité** : Le réseau pourrait prédire une **matrice nulle** au début de l'entraînement
→ Tous les points deviennent $(0, 0, 0)$ après transformation
→ Gradients nuls → **impossible d'apprendre** !

**Avec identité** : Au début, $M \approx I$ (transformation nulle = objet inchangé)
→ Le réseau apprend progressivement des ajustements
→ $M = I + \Delta M$ où $\Delta M$ est petit

**Analogie** : C'est comme commencer un dessin avec un croquis de base plutôt qu'une feuille blanche.

#### ÉTAPE 4 : Application de la Matrice

```python
# Dans PointNet forward
M = self.tnet(x)              # (B, 3, 3)
x_aligned = torch.bmm(M, x)   # Rotation des points
```

**`torch.bmm`** : Batch Matrix Multiplication

**Exemple concret** :

```python
# Points originaux (cylindre tourné)
x = [[0.5, 0.3, 0.7],
     [0.8, 0.6, 0.4],
     [0.2, 0.9, 0.1]]

# Matrice prédite
M = [[1.1,  0.05, -0.02],
     [0.03, 1.08,  0.01],
     [-0.01, 0.02, 1.09]]

# Points alignés : M × x
x_aligned = M @ x = [[0.54, 0.32, 0.77],
                     [0.86, 0.65, 0.44],
                     [0.21, 0.97, 0.11]]
```

**Résultat** : L'objet est maintenant dans une **orientation canonique** (toujours la même), ce qui facilite la classification.

---

## 🧠 PARTIE 4 : POINTNET COMPLET

### Architecture Globale

```
Input (B, 3, N)
    ↓
┌─────────────────┐
│  T-Net (3D)     │ → Matrice 3×3
└─────────────────┘
    ↓
Alignement spatial: points × matrice
    ↓
Conv1d(3→64) → Conv1d(64→64)
    ↓
┌─────────────────┐
│  T-Net (64D)    │ → Matrice 64×64
└─────────────────┘
    ↓
Alignement features: features × matrice
    ↓
Conv1d(64→128) → Conv1d(128→1024)
    ↓
MaxPooling global → (B, 1024)
    ↓
FC(1024→512) → FC(512→256) → FC(256→3)
    ↓
LogSoftmax → Probabilités de classes
```

### Code du Forward

```python
def forward(self, x):
    # x: (B, 3, N) - Batch, Coordonnées xyz, Nombre de points
    
    # ========== ALIGNEMENT SPATIAL ==========
    tn1 = self.tnet1(x)           # (B, 3, 3)
    x = torch.bmm(tn1, x)         # (B, 3, N) - Points alignés
    
    # ========== EXTRACTION FEATURES BAS NIVEAU ==========
    x = F.relu(self.fc1(x))       # (B, 64, N)
    x = F.relu(self.fc2(x))       # (B, 64, N)
    
    # ========== ALIGNEMENT FEATURES ==========
    tn2 = self.tnet2(x)           # (B, 64, 64)
    x = torch.bmm(tn2, x)         # (B, 64, N) - Features alignées
    
    # ========== FEATURES HAUT NIVEAU ==========
    x = F.relu(self.fc3(x))       # (B, 128, N)
    x = F.relu(self.fc4(x))       # (B, 1024, N)
    
    # ========== GLOBAL FEATURE (AGREGATION) ==========
    x = torch.max(x, dim=2)[0]    # (B, 1024) - Feature globale
    
    # ========== CLASSIFICATION ==========
    x = F.relu(self.fc5(x))       # (B, 512)
    x = F.relu(self.fc6(x))       # (B, 256)
    x = self.logsoftmax(self.fc7(x))  # (B, 3) - Probabilités log
    
    return x, tn1, tn2
```

### Flux de Données Détaillé

| Étape | Shape | Opération |
|-------|-------|-----------|
| Input | (B, 3, N) | Nuage de points brut |
| T-Net1 | (B, 3, 3) | Matrice de rotation spatiale |
| Aligned | (B, 3, N) | Points dans orientation canonique |
| Conv 1-2 | (B, 64, N) | Features par point |
| T-Net2 | (B, 64, 64) | Matrice de rotation features |
| Aligned | (B, 64, N) | Features alignées |
| Conv 3-4 | (B, 1024, N) | Features hautes par point |
| MaxPool | (B, 1024) | **Feature globale unique** |
| FC 5-7 | (B, 3) | Scores de classes |

### Pourquoi 2 T-Nets ? ⚠️ QUESTION D'EXAMEN

**1. T-Net1 (Input : 3D)** : 
- Aligne les **points** dans l'espace 3D
- Rotation **géométrique** (rotation physique de l'objet)
- Matrice 3×3 agit sur les coordonnées (x, y, z)

**2. T-Net2 (Input : 64D)** :
- Aligne les **features** dans l'espace des caractéristiques
- Rotation **abstraite** (ajustement de la représentation interne)
- Matrice 64×64 agit sur les 64 features par point

**Analogie** :
- **T-Net1** = Tourner la statue physiquement pour la voir de face
- **T-Net2** = Ajuster comment on "perçoit" la statue (filtres mentaux)

**Bénéfice** : Double niveau d'invariance → robustesse accrue

---

## 🔒 PARTIE 5 : RÉGULARISATION T-NET

### Code de Régularisation

```python
def tnet_regularization(matrix):
    # matrix: (B, k, k) - Matrices prédites par T-Net
    
    I = torch.eye(matrix.size(1)).to(matrix.device)
    MMT = torch.bmm(matrix, matrix.transpose(2, 1))  # M × M^T
    loss = torch.norm(MMT - I, dim=(1, 2)).mean()
    return loss

# Dans la loss totale
reg1 = tnet_regularization(tn1)
reg2 = tnet_regularization(tn2)
loss = criterion(output, target) + 0.001 * (reg1 + reg2)
```

### Concept Mathématique

**Objectif** : Forcer la matrice $M$ à être **proche d'une rotation** (matrice orthogonale).

**Propriété d'une matrice orthogonale** : 

$$M M^T = I$$

Où $I$ est la matrice identité.

**Formule de la loss de régularisation** :

$$L_{\text{reg}} = ||M M^T - I||_F^2$$

Où $||\cdot||_F$ est la **norme de Frobenius** : $||A||_F = \sqrt{\sum_{i,j} A_{ij}^2}$

### Exemple Concret

```python
# Matrice prédite (non orthogonale)
M = [[1.5, 0.2, 0.1],
     [0.1, 1.4, 0.2],
     [0.2, 0.1, 1.6]]

# M × M^T
M_MT = [[2.30, 0.33, 0.42],
        [0.33, 2.01, 0.42],
        [0.42, 0.42, 2.62]]

# Matrice identité
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

# Différence
diff = M_MT - I = [[1.30, 0.33, 0.42],
                   [0.33, 1.01, 0.42],
                   [0.42, 0.42, 1.62]]

# Loss de Frobenius
loss = sqrt(1.30² + 0.33² + 0.42² + ... + 1.62²) = 2.15
```

**Interprétation** : Plus la loss est élevée, plus $M$ s'éloigne d'une rotation pure.

### Pourquoi cette Régularisation ? ⚠️ QUESTION D'EXAMEN

**Sans régularisation** :
- Le T-Net pourrait apprendre des transformations **dégénérées** (ex: mise à l'échelle extrême, cisaillement)
- Exemple : $M$ pourrait "écraser" tous les points sur un plan → perte d'information
- Instabilité numérique

**Avec régularisation** :
- Le réseau apprend des matrices **plus propres** (proches de rotations pures)
- Préserve les distances et les angles
- Meilleure généralisation

**Coefficient 0.001** : Équilibre entre la loss de classification et la régularisation

---

## 🎯 PARTIE 6 : ENTRAÎNEMENT AVEC BRUIT

### Code d'Augmentation

```python
def bruit(points, sigma):
    """Ajoute du bruit gaussien aux points"""
    bruit = torch.randn_like(points) * sigma
    return points + bruit

# Pendant l'entraînement
max_sigma = 0.1  # Bruit maximal de 10%
sigma = torch.rand(1).item() * max_sigma  # σ aléatoire [0, 0.1]
points = bruit(points, sigma)
```

### Exemple Concret

```python
# Points originaux
points = [[0.5, 0.8, 0.2],
          [0.3, 0.6, 0.9],
          [0.7, 0.4, 0.1]]

# Bruit gaussien (σ = 0.05)
noise = [[-0.02, 0.03, -0.01],
         [0.01, -0.02, 0.04],
         [0.03, -0.01, 0.02]]

# Points bruités
points_noisy = [[0.48, 0.83, 0.19],
                [0.31, 0.58, 0.94],
                [0.73, 0.39, 0.12]]
```

**Effet visuel** : Les points "vibrent" légèrement autour de leur position originale.

### Résultats Expérimentaux

| Niveau de bruit | Accuracy | Impact |
|----------------|----------|---------|
| **Sans bruit (σ=0)** | ~95% | Baseline |
| **Bruit faible (σ=0.05)** | ~92% | -3% |
| **Bruit moyen (σ=0.1)** | ~90% | -5% |
| **Bruit fort (σ=0.5)** | ~33% | Effondrement |

### Analyse

**Robustesse modérée** :
- PointNet résiste bien au bruit léger (≤10%)
- Performance se dégrade rapidement au-delà
- À 50% de bruit, classification aléatoire (33% avec 3 classes)

**Pourquoi cette sensibilité ?**
- Le MaxPool global capture les features **les plus activées**
- Le bruit fort peut **déplacer les maxima** → features incorrectes
- Pas de mécanisme de débruitage explicite

**Solutions dans PointNet++** :
- Échantillonnage hiérarchique (multi-échelle)
- Groupement local de points
- Meilleure robustesse au bruit

---

## 🎓 POINTS CLÉS POUR L'EXAMEN

### Questions de Compréhension Probables

**1. Qu'est-ce que PointNet ?**
- Réseau de neurones pour **nuages de points 3D**
- Invariant à l'**ordre** des points (MaxPool)
- Invariant aux **rotations** (T-Net)

**2. Pourquoi Conv1d avec kernel=1 ?**
- Traiter chaque point **indépendamment**
- Pas de notion de voisinage spatial (points désordonnés)
- Applique la même transformation à tous les points

**3. Rôle du MaxPooling global ?** ⚠️ CRUCIAL
- Agrège les features de tous les points en **1 vecteur global**
- **Invariant à l'ordre** : max([a, b, c]) = max([c, a, b])
- Résume l'objet entier en (B, 1024)

**4. Qu'est-ce que le T-Net ?**
- Sous-réseau qui prédit une **matrice de transformation**
- Aligne automatiquement les objets dans une orientation canonique
- 2 T-Nets : spatial (3×3) et features (64×64)

**5. Pourquoi ajouter l'identité dans le T-Net ?**
- Éviter les transformations dégénérées au début
- Initialisation stable (M ≈ I au départ)
- Permet l'apprentissage progressif

**6. Pourquoi régulariser le T-Net ?**
- Forcer $M$ à être **orthogonale** (rotation propre)
- Éviter transformations dégénérées (écrasement, cisaillement)
- Meilleure stabilité et généralisation

**7. Différence entre les 2 T-Nets ?**
- **T-Net1** : Aligne les points dans l'espace 3D (géométrique)
- **T-Net2** : Aligne les features dans l'espace 64D (abstrait)

**8. Limites de PointNet ?**
- Ne capture **pas les relations locales** entre points voisins
- Sensible au **bruit fort** (>10%)
- Résolu par **PointNet++** (échantillonnage hiérarchique)

**9. Pourquoi `sqrt(u)` pour échantillonner un disque ?**
- Compense l'augmentation de l'aire avec le rayon
- Assure une distribution **uniforme** sur le disque
- Sans sqrt : concentration au centre

### Blocs de Code à Maîtriser

1. **T-Net** :
   - Conv1d (kernel=1) × 3
   - MaxPool global
   - MLP pour prédire matrice
   - Ajout identité

2. **PointNet Forward** :
   - T-Net1 → alignement spatial
   - Conv features
   - T-Net2 → alignement features
   - MaxPool global
   - Classification

3. **Régularisation** :
   - Calcul de $M M^T$
   - Norme de Frobenius de $M M^T - I$

4. **Data augmentation** :
   - Bruit gaussien
   - Rotation aléatoire
   - Normalisation

### Formules Importantes

**Conv1d (kernel=1)** :

$$y_i = W x_i + b$$

Appliqué indépendamment à chaque point $x_i$.

**MaxPool global** :

$$f = \max_{i=1}^{N} \{h(x_i)\}$$

Où $h$ est la fonction de transformation (Conv1d + ReLU).

**T-Net regularization** :

$$L_{\text{reg}} = ||MM^T - I||_F^2 = \sum_{i,j} (MM^T - I)_{ij}^2$$

**Loss totale** :

$$L = L_{\text{classification}} + \lambda (L_{\text{reg1}} + L_{\text{reg2}})$$

Typiquement $\lambda = 0.001$.

**Échantillonnage uniforme sur disque** :

$$r = \sqrt{u_1}, \quad \theta = 2\pi u_2$$

Où $u_1, u_2 \sim \mathcal{U}(0, 1)$.

---

## 📋 RÉSUMÉ RAPIDE

### Pipeline Complet de PointNet

```
Input: Nuage de points (B, 3, N)
         ↓
    [T-Net1 3×3]
  Alignement spatial
         ↓
    Conv1d (3→64)
         ↓
    [T-Net2 64×64]
  Alignement features
         ↓
  Conv1d (64→1024)
         ↓
   [MaxPool Global]
  Feature globale (B, 1024)
         ↓
    MLP Classifier
         ↓
  Classes (B, 3)
```

### Comparaison avec les TPs Précédents

| Aspect | TP1 (Neural Prior) | TP2 (Style Transfer) | TP3 (SE Block) | TP4 (PointNet) |
|--------|-------------------|---------------------|----------------|----------------|
| **Type de données** | Images 2D | Images 2D | Images 2D (CIFAR) | Nuages points 3D |
| **Optimisé** | Poids réseau | Pixels | Poids réseau | Poids réseau |
| **Mécanisme clé** | Prior implicite | Gram matrix | Channel attention | MaxPool + T-Net |
| **Invariance** | Aucune | Aucune | Aucune | Ordre + Rotation |
| **Dataset** | 1 image | 2 images | 50k images | Objets 3D |
| **But** | Reconstruction | Style transfer | Classification | Classification 3D |

---

## 💡 ASTUCES POUR L'EXAMEN

### Concepts à Bien Comprendre

1. **Pourquoi kernel=1 ?** → Points sans structure spatiale fixe
2. **Rôle du MaxPool** → Invariance à l'ordre (permutation)
3. **Identité dans T-Net** → Stabilité d'entraînement
4. **2 T-Nets** → Double niveau d'alignement (spatial + features)
5. **Régularisation** → Matrices orthogonales (rotations pures)
6. **sqrt(u) pour disque** → Distribution uniforme

### Schémas à Retenir

**Architecture T-Net** :
```
Points (3, N)
    ↓
Conv1d (kernel=1)
    ↓
MaxPool global
    ↓
MLP
    ↓
+ Identity
    ↓
Matrice (3, 3)
```

**Propriété MaxPool** :
```
Points: [p1, p2, p3]
Features: [f(p1), f(p2), f(p3)]
MaxPool: max{f(p1), f(p2), f(p3)}

Peu importe l'ordre !
```

---

**Voilà ! Tu as maintenant une compréhension complète du TP4 - PointNet** 🎉

**Résumé en 3 mots** : **Points → Features → Classes** !

**Points clés** :
- ✅ Conv1d kernel=1 pour points indépendants
- ✅ MaxPool global pour invariance à l'ordre
- ✅ T-Net pour invariance aux rotations
- ✅ Régularisation pour matrices orthogonales

**Bon courage pour ton examen !** 🚀
