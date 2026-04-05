'''
Kernel Support Vector Machine (SVM) on YelpReviewFull dataset (5-class star rating prediction).
Anthony Mahon - CAP5610 Spring 2026

Run modes:
    python Kernel_SVM.py
'''


from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from utils import timed_step


def main():
    #Load and sub-sample the data
    with timed_step("Loading Yelp dataset (sub-sampled)"):
        dataset = load_dataset("yelp_review_full")

        #Take a small subset for training and testing
        train_data = dataset['train'].select(range(10000))
        test_data = dataset['test'].select(range(2000))

        X_train_text = train_data['text']
        y_train = train_data['label']

        X_test_text = test_data['text']
        y_test = test_data['label']

    #Vectorize the input. We are going to scale and normalize to prevent distance distortion
    with timed_step("Applying TF-IDF Vectorization"):
        #max_features limits the dictionary to the top 10,000 most important words
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

        #Learn the vocabulary and transform the text into numbers
        X_train_vectors = vectorizer.fit_transform(X_train_text)

        #Only transform, not fit, the test data
        X_test_vectors = vectorizer.transform(X_test_text)

    #Initialize and Train the Kernel SVM :)
    with timed_step("Training Kernel SVM (RBF)"):
        #kernel='rbf' is the "Radial Basis Function" (the Kernel trick)
        #C=1.0 is the default penalty for mistakes
        svm_model = SVC(kernel='rbf', C=1.0)

        #This is where the learning happens
        svm_model.fit(X_train_vectors, y_train)

    #Predict and Evaluate
    with timed_step("Making predictions and evaluating"):
        y_pred = svm_model.predict(X_test_vectors)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n--- Results ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("\nDetailed Classification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]))

if __name__ == "__main__":
    main()
