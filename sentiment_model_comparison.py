import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle  # <-- added


# Load dataset
df = pd.read_csv('hr_sentiment_dataset.csv')

# Clean sentences: remove special characters
def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text))

df['sentence'] = df['sentence'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['label'], test_size=0.2, random_state=0)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save the fitted vectorizer so the Flask app can load it
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


models = [
    (LogisticRegression(max_iter=1000), 'Logistic Regression'),
    (LinearSVC(), 'SVM'),
    (RandomForestClassifier(), 'Random Forest'),
    (MultinomialNB(), 'Naive Bayes'),
    (KNeighborsClassifier(), 'KNN'),
    (DecisionTreeClassifier(), 'Decision Tree'),
]

fitted_models = {}
for model, name in models:
    try:
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, pred)
        print(f'{name} Accuracy:', acc)
        fitted_models[name] = model
        # Export Random Forest model after fitting
        if name == 'Random Forest':
            with open('random_forest_sentiment_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    except Exception as e:
        print(f'{name} failed: {e}')


# --- User input prediction section ---

def print_sentiment(model, model_name, user_vec):
    pred = model.predict(user_vec)[0]
    # Try to get confidence if possible
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(user_vec)[0]
        confidence = max(proba)
        print(f"{model_name} prediction: {pred} (confidence: {confidence:.2f})")
    elif hasattr(model, 'decision_function'):
        # For models like LinearSVC, use decision_function as a proxy
        try:
            decision = model.decision_function(user_vec)[0]
            import math
            confidence = 1 / (1 + math.exp(-decision)) if hasattr(model, 'classes_') and len(model.classes_) == 2 else float('nan')
            print(f"{model_name} prediction: {pred} (confidence: {confidence:.2f})")
        except Exception:
            print(f"{model_name} prediction: {pred} (confidence: N/A)")
    else:
        print(f"{model_name} prediction: {pred} (confidence: N/A)")

while True:
    user_sentence = input("\nEnter a sentence to analyze sentiment: ")
    user_sentence_clean = clean_text(user_sentence)
    user_vec = vectorizer.transform([user_sentence_clean])
    print("\nSentiment predictions for your input:")
    for name, model in fitted_models.items():
        print_sentiment(model, name, user_vec)
    again = input("\nDo you want to test another sentence? (y/n): ").strip().lower()
    if again not in ("y", "yes"):
        print("Exiting sentiment tester.")
        break
