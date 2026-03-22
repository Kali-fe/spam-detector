"""
train_model.py — Entraîne un modèle de détection de spam et le sauvegarde.
Lance ce fichier une seule fois avant de démarrer l'app Flask.
"""

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Dataset d'entraînement ────────────────────────────────────────────────────
# Format : (texte, label)  label = 1 → spam, 0 → ham (normal)
DATA = [
    # SPAM
    ("Félicitations ! Vous avez gagné 1 000 000 €. Cliquez ici pour réclamer votre prix.", 1),
    ("OFFRE LIMITÉE : Achetez maintenant et obtenez 90% de réduction !", 1),
    ("Vous avez été sélectionné pour recevoir un iPhone gratuit. Répondez maintenant !", 1),
    ("Gagnez de l'argent facilement depuis chez vous. Inscription gratuite.", 1),
    ("URGENT : Votre compte bancaire a été compromis. Cliquez ici immédiatement.", 1),
    ("Free Viagra pills - click here now!", 1),
    ("Congratulations! You are our lucky winner. Claim your $500 gift card.", 1),
    ("Make money fast! Work from home, earn $5000 per week guaranteed.", 1),
    ("Dear friend, I am a Nigerian prince and I need your help to transfer $10 million.", 1),
    ("Hot singles in your area! Click to meet them now.", 1),
    ("You have won a free cruise! Call now to claim your prize.", 1),
    ("Buy cheap medication online, no prescription needed!", 1),
    ("Lose 30 pounds in 30 days with this miracle pill!", 1),
    ("Your PayPal account has been suspended. Click here to restore access.", 1),
    ("Limited time offer: Get rich quick with this secret investment strategy.", 1),
    ("Click here to claim your free gift card worth $1000!", 1),
    ("Earn $500 per day from home! No experience needed!", 1),
    ("WINNER WINNER! You've been randomly selected for our prize draw.", 1),
    ("Act now! This offer expires in 24 hours. Don't miss out!", 1),
    ("Get a free laptop just for completing this survey!", 1),
    ("Votre livraison est en attente. Payez 2€ de frais pour recevoir votre colis.", 1),
    ("Votre abonnement Netflix va expirer. Cliquez ici pour renouveler.", 1),
    ("Vous avez un remboursement fiscal en attente. Saisissez vos coordonnées.", 1),
    ("PROMO FLASH : -70% sur tout le site. Commandez maintenant !", 1),
    ("Investissez en crypto et doublez votre mise en 24h. Garantie.", 1),
    ("Obtenez un prêt de 50 000€ sans justificatif. Réponse immédiate.", 1),
    ("Vous avez été piraté ! Changez votre mot de passe en cliquant ici.", 1),
    ("Rencontrez des célibataires près de chez vous. Inscription 100% gratuite.", 1),
    ("Votre PC est infecté par un virus ! Appelez notre support au 0800 000 000.", 1),
    ("Gagnez des bitcoins gratuitement en 5 minutes par jour !", 1),
    # HAM
    ("Bonjour, est-ce qu'on peut se retrouver demain pour déjeuner ?", 0),
    ("N'oublie pas la réunion de 14h cet après-midi.", 0),
    ("Merci pour ton aide sur le projet, c'était vraiment utile.", 0),
    ("Est-ce que tu peux me rappeler quand tu as un moment ?", 0),
    ("Je t'envoie le rapport dès que je l'ai terminé.", 0),
    ("Hey, tu as vu le match hier soir ? Incroyable !", 0),
    ("Bonne anniversaire ! J'espère que tu passes une excellente journée.", 0),
    ("La réunion de demain est reportée à jeudi 10h.", 0),
    ("Peux-tu relire mon document avant que je l'envoie ?", 0),
    ("J'ai réservé un restaurant pour samedi soir, ça te va ?", 0),
    ("Hello, can we schedule a call for tomorrow morning?", 0),
    ("Just wanted to check in and see how you're doing.", 0),
    ("The meeting has been moved to 3 PM. Please update your calendar.", 0),
    ("Thanks for sending over the report. I'll review it tonight.", 0),
    ("Can you pick up some groceries on your way home?", 0),
    ("Happy birthday! Hope you have a wonderful day.", 0),
    ("Let me know when you're free to chat about the project.", 0),
    ("I've attached the presentation slides for your review.", 0),
    ("Don't forget we have dinner at mom's place this Sunday.", 0),
    ("Looking forward to seeing you at the conference next week.", 0),
    ("Salut, tu peux m'expliquer comment fonctionne cette fonctionnalité ?", 0),
    ("Le déjeuner de vendredi est confirmé. On se retrouve à midi.", 0),
    ("J'ai terminé les modifications sur le site. Peux-tu vérifier ?", 0),
    ("Rappelle-moi d'appeler le médecin ce soir.", 0),
    ("Le train est en retard de 20 minutes. Je serai là vers 18h.", 0),
    ("Ton colis a été expédié et sera livré dans 3 à 5 jours ouvrables.", 0),
    ("Votre rendez-vous du 25 mars à 10h est bien confirmé.", 0),
    ("Nous avons bien reçu votre demande et reviendrons vers vous sous 48h.", 0),
    ("Le rapport mensuel est disponible dans votre espace personnel.", 0),
    ("Votre facture du mois de mars est disponible en téléchargement.", 0),
]

texts  = [d[0] for d in DATA]
labels = [d[1] for d in DATA]

# ── Entraînement ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words=None,          # on garde les mots car bilingue
        sublinear_tf=True,
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

pipeline.fit(X_train, y_train)

# ── Évaluation ────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("=== Résultats sur le jeu de test ===")
print(classification_report(y_test, y_pred, target_names=["Ham (normal)", "Spam"]))

# ── Sauvegarde ────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Modèle sauvegardé dans model/spam_model.pkl")
