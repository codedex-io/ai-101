from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

texts = [
  "I love programming.", "Python is amazing.", "I enjoy machine learning.",
  "The weather is nice today.", "I like algo.", "Machine learning is fascinating.",
  "Natural Language Processing is a part of AI."
]
labels = [
  "tech", "tech", "tech", "non-tech", "tech", "tech", "tech"
]

# Convert labels to numeric format
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(labels)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels_numeric, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
