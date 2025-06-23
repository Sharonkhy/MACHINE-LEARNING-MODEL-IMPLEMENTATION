# %% [markdown]
# # Spam Email Detection using Machine Learning
# 
# This notebook demonstrates the implementation of a predictive model to classify emails as spam or ham (not spam) using scikit-learn.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# %%
# Load the dataset
df = pd.read_csv('spam.csv', sep='\t', names=['label', 'text'], encoding='latin-1')
df.head()

# %%
# Data cleaning and preprocessing
# Drop unnecessary columns and rename the remaining ones
df = df[['label', 'text']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# %%
# Exploratory Data Analysis
print("Class distribution:\n", df['label'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam vs Ham')
plt.xlabel('Label (0=Ham, 1=Spam)')
plt.ylabel('Count')
plt.show()

# %%
# Text preprocessing function
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stemmer = PorterStemmer()

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords and stem
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# %%
# Split the data into training and testing sets
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create a pipeline with TF-IDF vectorizer and classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Define parameter grid for GridSearchCV
parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
    'clf__alpha': (1e-2, 1e-3)
}

# Perform grid search
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# %%
# Print best parameters
print("Best parameters found:")
print(grid_search.best_params_)

# %%
# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
# Compare with other models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear')
}

for name, model in models.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 2))),
        ('clf', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# %%
# Example predictions
test_emails = [
    "Congratulations! You've won a $1000 gift card. Click here to claim!",
    "Hi John, just checking in about our meeting tomorrow at 2pm.",
    "Your account statement is ready. Please find attached.",
    "URGENT: Your bank account has been compromised. Verify your details now!"
]

for email in test_emails:
    prob = best_model.predict_proba([email])[0]
    prediction = best_model.predict([email])[0]
    print(f"\nEmail: {email}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
    print(f"Probability (Ham: {prob[0]:.4f}, Spam: {prob[1]:.4f})")

# %%
# Save the model for future use
import joblib
joblib.dump(best_model, 'spam_classifier.joblib')

# %%
# Load and test the saved model
loaded_model = joblib.load('spam_classifier.joblib')
new_email = "Free viagra now!!! Limited offer!!!"
prediction = loaded_model.predict([new_email])[0]
print(f"\nNew email: {new_email}")
print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
