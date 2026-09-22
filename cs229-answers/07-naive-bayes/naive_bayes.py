import collections
import numpy as np
import util
import svm


def get_words(message):
    return message.lower().split()


def create_dictionary(messages):
    """Keep words with at least five occurrences in the training messages."""
    words = [word for message in messages for word in get_words(message)]

    counter = collections.Counter(words)
    valid = [word for word, count in counter.items() if count >= 5]
    return {valid[i] : i for i in range(len(valid))}


def transform_text(messages, word_dictionary):
    """Build a message-by-word count matrix, ignoring unknown words."""
    n, m = len(messages), len(word_dictionary)
    freqs = np.zeros((n,m), dtype=int)

    for i in range(n):
        words = get_words(messages[i])
        for word in words:
            if word in word_dictionary:
                freqs[i][word_dictionary[word]] += 1

    return freqs


def fit_naive_bayes_model(matrix, labels):
    """Estimate class priors and word probabilities with Laplace smoothing."""
    m, n = matrix.shape

    phi_y = np.mean(labels)
    phi_j_y1 = (1 + matrix[labels==1].sum(axis=0)) / (n + np.sum(matrix[labels==1]))
    phi_j_y0 = (1 + matrix[labels==0].sum(axis=0)) / (n + np.sum(matrix[labels==0]))

    return phi_y, phi_j_y1, phi_j_y0


def predict_from_naive_bayes_model(model, matrix):
    """Classify using the log posterior odds."""
    phi_y, phi_j_y1, phi_j_y0 = model

    return (matrix @ (np.log(phi_j_y1)-np.log(phi_j_y0)) + np.log(phi_y) - np.log(1-phi_y)) >= 0


def get_top_five_naive_bayes_words(model, dictionary):
    """Return the five words with the largest spam log-likelihood ratios."""
    phi_y, phi_j_y1, phi_j_y0 = model

    newd = {v : k for k,v in dictionary.items()}

    metric = np.argsort(np.log(phi_j_y1)-np.log(phi_j_y0))[-5:]

    return [newd[i] for i in metric]


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Choose the RBF radius with the highest validation accuracy."""
    r_best = (radius_to_consider[0], 0)

    for r in radius_to_consider:
        output = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, r)
        score = np.sum(val_labels==output)

        if score > r_best[1]:
            r_best = (r, score)

    return r_best[0]


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    train_messages, train_labels = util.load_spam_dataset('data/ds6_train.tsv')

    val_messages, val_labels = util.load_spam_dataset('data/ds6_val.tsv')

    test_messages, test_labels = util.load_spam_dataset('data/ds6_test.tsv')

    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    val_matrix = transform_text(val_messages, dictionary)

    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    print(naive_bayes_model)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

    train_matrix, train_labels
