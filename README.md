# 🛡️ SpamRadar — Détecteur de Spam par ML

Application web de détection de spam utilisant **Python, Flask et Scikit-learn**.

## 🗂️ Structure du projet

```
spam_detector/
├── app.py              # Serveur Flask
├── train_model.py      # Script d'entraînement du modèle ML
├── requirements.txt    # Dépendances Python
├── Procfile            # Pour déploiement (Render/Railway)
├── render.yaml         # Config Render
├── model/
│   └── spam_model.pkl  # Modèle sauvegardé (généré par train_model.py)
└── templates/
    └── index.html      # Interface web
```

---

## 🚀 Lancer en local

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle (une seule fois)

```bash
python train_model.py
```

### 3. Lancer l'application

```bash
python app.py
```

Ouvre http://localhost:5000 dans ton navigateur.

---

## ☁️ Déployer sur Render (gratuit)

1. Crée un compte sur [render.com](https://render.com)
2. **New → Web Service → Connect GitHub repo**
3. Sélectionne ton repo
4. Render détecte automatiquement `render.yaml`
5. Clique **Deploy** — c'est tout !

Le `buildCommand` dans `render.yaml` installe les dépendances **et** entraîne le modèle automatiquement.

---

## ☁️ Déployer sur Railway

```bash
# Installe Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
```

---

## 🧠 Comment ça marche

| Étape | Technologie | Détail |
|-------|-------------|--------|
| Vectorisation | TF-IDF (bigrammes) | Transforme le texte en vecteurs numériques |
| Classification | Naïve Bayes multinomial | Prédit spam (1) ou normal (0) |
| API | Flask `/predict` | Reçoit le texte, retourne JSON |
| Interface | HTML/CSS/JS vanilla | Affiche le résultat + barres de confiance |

---

## 🎯 Résultat attendu

- **Précision** : ~90–95% sur le jeu de test inclus
- **Langues** : Français et Anglais
- **Latence** : < 50 ms par requête
