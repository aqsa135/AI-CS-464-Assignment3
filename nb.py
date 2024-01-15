# ## Naive Bayes using NLTK
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from math import log
from random import shuffle
from filters import *


def train(training_set, filters, transforms):
    model = ConditionalFreqDist()
    for category in training_set:
        files = training_set[category]
        for name in files:
            word_list = apply_transforms(transforms, select_features(filters, movie_reviews.words(name)))
            for word in word_list:
                model[category][word] += 1
    return model


def classify(model, list_of_tokens):
    filtered_tokens = apply_transforms(transforms, select_features(filters, list_of_tokens))
    categories = list(model.keys())
    results = {}
    vocabulary_size = len(set([word for category in categories for word in model[category].keys()]))

    for category in categories:
        total_words_in_category = sum(model[category].values())
        log_likelihood = log(0.5)  # I am assuming uniform priors

        for word in filtered_tokens:
            word_frequency = model[category][word] + 1
            prob_word_given_category = word_frequency / (total_words_in_category + vocabulary_size)
            log_likelihood += log(prob_word_given_category)

        results[category] = log_likelihood

    return results


if __name__ == "__main__":
    fileids = movie_reviews.fileids()
    shuffle(fileids)
    partition_size = len(fileids) // 5
    partitions = [fileids[i:i + partition_size] for i in range(0, len(fileids), partition_size)]

    # Testing with filters/transforms
    accuracies_with_filters = []
    for i in range(5):
        test_set = partitions[i]
        training_set = [fileid for j, partition in enumerate(partitions) if j != i for fileid in partition]
        tset = {'pos': [fileid for fileid in training_set if fileid.startswith('pos')],
                'neg': [fileid for fileid in training_set if fileid.startswith('neg')]}

        model = train(tset, filters, transforms)
        correct = 0
        for fileid in test_set:
            result = classify(model, movie_reviews.words(fileid))
            true_val = 'pos' if fileid.startswith('pos') else 'neg'
            predicted = 'pos' if result['pos'] > result['neg'] else 'neg'
            if predicted == true_val:
                correct += 1
        accuracy = correct / len(test_set)
        accuracies_with_filters.append(accuracy)
        print(f"Fold {i+1} accuracy with filters: {accuracy:.4f}")

    average_accuracy_with_filters = sum(accuracies_with_filters) / 5
    print(f"\nAverage accuracy with filters over 5 folds: {average_accuracy_with_filters:.4f}")

    # Testing without filters/transforms
    accuracies_without_filters = []
    for i in range(5):
        test_set = partitions[i]
        training_set = [fileid for j, partition in enumerate(partitions) if j != i for fileid in partition]
        tset = {'pos': [fileid for fileid in training_set if fileid.startswith('pos')],
                'neg': [fileid for fileid in training_set if fileid.startswith('neg')]}

        model = train(tset, [], [])  # Passing empty lists for filters and transforms
        correct = 0
        for fileid in test_set:
            result = classify(model, movie_reviews.words(fileid))
            true_val = 'pos' if fileid.startswith('pos') else 'neg'
            predicted = 'pos' if result['pos'] > result['neg'] else 'neg'
            if predicted == true_val:
                correct += 1
        accuracy = correct / len(test_set)
        accuracies_without_filters.append(accuracy)
        print(f"Fold {i+1} accuracy without filters: {accuracy:.4f}")

    average_accuracy_without_filters = sum(accuracies_without_filters) / 5
    print(f"\nAverage accuracy without filters over 5 folds: {average_accuracy_without_filters:.4f}")
