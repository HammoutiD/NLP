import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs


def preproc(text: str) -> list[str]:
    # Your code here
    # return a pre-processed list of tokens with the pipeline you designed
    pass


if __name__ == "__main__":
    data = pd.read_csv("./data/spam.tsv", sep="\t", names=["label", "text"], header=None, encoding='utf-8')
    # print(data.head())

    X = data[['text']]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(analyzer=preproc, lowercase=False)
    tfidf_vect = vectorizer.fit(X_train['text'])

    tfidf_train = tfidf_vect.transform(X_train['text'])
    tfidf_test = tfidf_vect.transform(X_test['text'])

    classifier = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)
    model = classifier.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)

    precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

    print('Accuracy: {}\nF1-score: {}\nPrecision: {}\nRecall: {}'.format(
        round(acs(y_test, y_pred) * 100, 3), round(fscore, 3), round(precision, 3), round(recall, 3)))
