# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. UČITAJ PODATKE
df = pd.read_csv('products.csv')

# 2. IZVUCI POTREBNE KOLONE (uredi i očisti)
df = df[['Product Title', ' Category Label']].dropna()
df.columns = ['title', 'category']

# 3. VEKTORIZACIJA
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['title'])
y = df['category']

# 4. PODELA NA TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRENIRANJE MODELA
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. EVALUACIJA
y_pred = model.predict(X_test)
print("Izveštaj o tačnosti modela:")
print(classification_report(y_test, y_pred))

# 7. SACUVAMO MODEL I VEKTORIZER
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model i vektorizer su uspešno sačuvani.")

# predict_category.py

import joblib

# Učitaj sačuvane modele
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_category(title):
    # Pretvori unos u vektore
    title_vector = vectorizer.transform([title])
    # Predvidi kategoriju
    prediction = model.predict(title_vector)
    return prediction[0]

if __name__ == "__main__":
    print("Unesite naziv proizvoda (npr. 'iphone 7 32gb gold'):")
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ['exit', 'quit', 'kraj']:
            break
        category = predict_category(user_input)
        print(f"Predviđena kategorija: {category}\n")
