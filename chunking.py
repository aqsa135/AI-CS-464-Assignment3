import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from nltk import ne_chunk


# ## In this question, you'll get a little bit of practice using the NLTK chunker to extract
# ## structured knowledge.
# ## This function will take a string representing a sentence and give us back a list of tagged tokens.

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)

# # example = 'Tens of thousands marched as protests and strikes against unpopular pension reforms gripped France again Tuesday, with police ramping up security after the government warned that radical demonstrators intended to destroy, to injure and to kill.'
# New code for processing the five sentences
sentences = [
    "Apple announced its latest iPhone model at the California headquarters yesterday.",
    "The UN has expressed concerns over the humanitarian situation in Yemen.",
    "NASA's Perseverance rover landed on Mars, marking a significant achievement in space exploration.",
    "Bill Gates and Melinda French Gates have announced their decision to end their marriage after 27 years.",
    "Tesla's share price soared after its recent successful launch of a new electric truck."
]


## This piece of code builds a Regex parser to recognize noun phrases.
# ## It says that a noun phrase is 0 or 1 determiners (a, an, the), followed by 0 or more adjectives (JJ),
# ## followed by a noun.
# ## Run it on your examples. Does it get them right? Are there cases in which a noun phrase was not recognized?
# ## why do you think that is?

for sent in sentences:
    print("\nOriginal Sentence:", sent)

    # Regex parser for noun phrases
    s = preprocess(sent)
    cs = cp.parse(s)
    print("\nNoun Phrases using Regex parser:")
    print(cs)

    # Named entity chunker
    ne_tree = ne_chunk(pos_tag(word_tokenize(sent)))
    print("\nNamed Entities:")
    print(ne_tree)
 ## NLTK also contains its own built-in tool for recognizing named entities.
 ## Note that, in the tree, enities are tagged with tags such as GPE (geo-political entity),
 # Organization, or Person. https://www.nltk.org/book/ch07.html lists all of the types.



# Correct extraction: "iPhone", "model", "headquarters", "yesterday
# that were missed from the first sentence:  "latest iPhone model", "California headquarters"
# it identifies single nouns better compared to the compound noun

#Name entity
# Correct: "Apple" (as PERSON), "iPhone" (as ORGANIZATION), "California" (as GPE)
# Missed:  "iPhone" shouldn't be categorized as ORGANIZATION, and "Apple" is a company, so it might be better classified as ORGANIZATION.
# tends to properly recognize single proper nouns. chunker sometimes misclassifies entities, e.g., "Mars" as PERSON or "iPhone" as ORGANIZATION