
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Global variables
df = None
dataset = None
x_train, x_test, y_train, y_test = None, None, None, None
vectorizer = None
model = None
df1 = None

# ------------------------------
# Basic Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/drug')
def drug():
    return render_template('drug.html')

# ------------------------------
# Data Upload and View
# ------------------------------
@app.route('/load', methods=["GET", "POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        if not data.filename.endswith('.csv'):
            msg = "Invalid file format. Please upload a CSV file."
            return render_template('load.html', msg=msg)
        df = pd.read_csv(data, on_bad_lines='skip')
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

# ------------------------------
# Preprocessing
# ------------------------------
def review_clean(review): 
    # Convert to lowercase
    lower = review.str.lower()
    # Replace repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    # Remove special characters (using regex=True)
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]', ' ', regex=True)
    # Remove non-ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+', ' ', regex=True)
    # Trim whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$', '', regex=True)
    # Replace multiple spaces with a single space
    multiw_remove = whitespace_remove.str.replace(r'\s+', ' ', regex=True)
    # Replace two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ', regex=True)
    return dataframe

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test, vectorizer, df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100

        df = df.drop(columns=['uniqueID', 'date'])

        # Convert 'rating' to numeric and handle NaNs
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'].fillna(0, inplace=True)

        # Create a sentiment column based on rating
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x > 5 else 'negative')
        le = LabelEncoder()
        df['sentiment'] = le.fit_transform(df['sentiment'])

        # Clean the reviews
        df['review'] = review_clean(df['review'])
        df['review'].fillna('', inplace=True)

        # Define feature and target variables
        x = df['review']
        y = df['sentiment']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=size, random_state=42)

        # Vectorize the text data
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and Splitted Successfully')
    return render_template('preprocess.html')

# ------------------------------
# Model Training (Static Metrics)
# ------------------------------
@app.route('/model', methods=['POST', 'GET'])
def model():
    global model
    if request.method == "POST":
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            model = LogisticRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            msg = f'The accuracy obtained by Logistic Regression is {accuracy:.2f}%'
            return render_template('model.html', msg=msg)
        elif s == 2:
            model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            msg = f'The accuracy obtained by Decision Tree Classifier is {accuracy:.2f}%'
            return render_template('model.html', msg=msg)
        elif s == 3:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            msg = f'The accuracy obtained by Random Forest Classifier is {accuracy:.2f}%'
            return render_template('model.html', msg=msg)
        elif s == 4:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            msg = f'The accuracy obtained by XGBoost Classifier is {accuracy:.2f}%'
            return render_template('model.html', msg=msg)
    return render_template('model.html')

# ------------------------------
# Drug Recommendation Function
# ------------------------------
def predict_top_k_drugs(condition, k=3, alpha=0.5):
    global model, vectorizer, df
    """
    Given a condition, filter the dataset to find relevant drug reviews, compute a weighted sentiment score,
    and return the top k drugs.
    """
    # Filter the dataset by condition (assuming df1 is the dataset with additional columns)
    condition_data = df[df['condition'].str.contains(condition, case=False, na=False)].copy()
    if condition_data.empty:
        return "No matching condition found in the dataset."

    # Vectorize reviews and get sentiment probabilities using the trained RandomForest model (rf_model)
    review_vectors = vectorizer.transform(condition_data['review'])
    # Use the model trained on df1 data; ensure you have trained it as rf_model
    sentiment_probs = model.predict_proba(review_vectors)[:, 1]  # Probability of being positive

    # Normalize rating and usefulCount
    max_rating = condition_data['rating'].max()
    max_useful = condition_data['usefulCount'].max()
    condition_data['normalized_rating'] = condition_data['rating'] / max_rating if max_rating > 0 else 1
    condition_data['normalized_useful'] = condition_data['usefulCount'] / max_useful if max_useful > 0 else 1

    # Compute weighted sentiment score
    condition_data['weighted_sentiment'] = sentiment_probs * (
        alpha * condition_data['normalized_rating'] + (1 - alpha) * condition_data['normalized_useful']
    )

    # Get top K drugs based on the weighted sentiment score
    top_k_drugs = (
        condition_data.groupby('drugName')['weighted_sentiment']
        .mean()
        .nlargest(k)
        .index.tolist()
    )
    return top_k_drugs

# ------------------------------
# Prediction / Recommendation Route
# ------------------------------
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        # Get the condition from the form (updated from 'review' to 'condition')
        condition_input = request.form.get('condition')
        if not condition_input:
            msg = "Please enter a medical condition."
            return render_template('prediction.html', sentiment="", drug_name="", condition="", recommended_drugs=[], msg=msg)
        top_k_drugs = predict_top_k_drugs(condition_input, k=5, alpha=0.7)
        # For demonstration, we assume sentiment is 'Positive' if recommendations are returned
        sentiment = "Positive" if isinstance(top_k_drugs, list) else "N/A"
        drug_name = ", ".join(top_k_drugs) if isinstance(top_k_drugs, list) else top_k_drugs
        return render_template('prediction.html', sentiment=sentiment, drug_name=drug_name, condition=condition_input, recommended_drugs=top_k_drugs, msg="Prediction complete.")
    return render_template('prediction.html', sentiment="", drug_name="", condition="", recommended_drugs=[], msg="")

if __name__ == '__main__':
    app.run(debug=True)