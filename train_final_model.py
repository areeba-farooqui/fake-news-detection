import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the merged, balanced dataset
df = pd.read_csv("news_final_balanced.csv")

# Clean any missing data
df = df.dropna()

# Convert labels if needed (FAKE = 0, REAL = 1)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Features and labels
X = df['text']
y = df['label']

# Text processing
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model retrained. Test accuracy: {accuracy:.2f}")
