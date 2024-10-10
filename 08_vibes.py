from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from textblob import TextBlob

# Sample movie reviews
reviews = [
    "This movie was fantastic! Amazing, iconic",
    "I loved it!", 
    "Amazing storyline and great acting!",
    "The plot was cringe.",
    "Loved the acting! Highly recommended."
]

# Labels for the reviews
labels = ["positive", "positive", "positive", "negative", "positive"]

# (Optional) Correct any spelling mistakes in the reviews using TextBlob
corrected_reviews = [str(TextBlob(review).correct()) for review in reviews]

# Convert the text data into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corrected_reviews)

# Create a Naive Bayes classifier
model = MultinomialNB()

# Perform cross-validation (5-fold in this case)
scores = cross_val_score(model, X, labels, cv=5)

# Calculate and print the average accuracy
print(f"Cross-validated Accuracy: {scores.mean()}")

# Additional message based on average accuracy
if scores.mean() > 0.5:
    print("The vibes are great, book the tickets!")
else:
    print("The vibes are iffy")
