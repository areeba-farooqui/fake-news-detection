import pickle

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Test with a very obviously fake news
text = ["Aliens landed in Delhi and offered samosas to the Prime Minister."]

# Vectorize the input
X = vectorizer.transform(text)

# Get prediction and probabilities
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0]

# Output results
print("Predicted Label:", pred)
print(f"Prediction Probabilities → FAKE: {proba[0]:.4f}, REAL: {proba[1]:.4f}")
