#Aqsa Noreen
import nltk
import random
from nltk.book import *
from nltk.corpus import movie_reviews, stopwords
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist

# part1
texts = [text1, text2, text3, text7]
words = ['great', 'king', 'country', 'fear', 'love']

for i in texts:
    print("\nFor {}:".format(i.name))
    for word in words:
        print("\nWords similar to {} in {}:".format(word, i.name))
        i.similar(word)

# part2
for i in texts:
    randTok = random.sample(list(i), 50)
    print("\n50 random tokens from {}:".format(i.name))
    print(' '.join(randTok))

# part3/4
nltk.download('movie_reviews')
nltk.download('stopwords')
# Set
s_words = set(stopwords.words('english'))
# Extract words/ cleaning
pos_words = [word.lower() for word in movie_reviews.words(categories='pos') if word.isalpha() and word.lower() not in s_words]
neg_words = [word.lower() for word in movie_reviews.words(categories='neg') if word.isalpha() and word.lower() not in s_words]
# Frequency Distributions
pos_freq = FreqDist(pos_words)
neg_freq = FreqDist(neg_words)
# 10 most common words
print("10 most common words in positive reviews:")
for w, freq in pos_freq.most_common(10):
    print(f"{w}: {freq}")

print("\n10 most common words in negative reviews:")
for w, freq in neg_freq.most_common(10):
    print(f"{w}: {freq}")

#Does that help to distinguish between positive and negative reviews?
# yes it did help distinguish between the two because words like bad is more common in negative reviews

#Part 5
condfreqdist = ConditionalFreqDist(
    (category, word.lower())
    for category in movie_reviews.categories()
    for word in movie_reviews.words(categories=category)
    if word.isalpha() and word.lower() not in s_words
)

# 10 most common words for each category
for category in movie_reviews.categories():
    print(f"\n10 most common words in {category} reviews:")
    for w, freq in condfreqdist[category].most_common(10):
        print(f"{w}: {freq}")


# work cited
# https://notebook.community/Mashimo/datascience/03-NLP/introNLTK
# https://www.nltk.org/api/nltk.probability.ConditionalFreqDist.html#:~:text=Conditional%20frequency%20distributions%20are%20used,a%20document%2C%20given%20its%20length.
