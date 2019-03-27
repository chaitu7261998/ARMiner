# Use python 3 or above

import re
import numpy as np

# Returns preprocessed numpy array
def preprocess_input(filename_str):

    # Ignore these words
    stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
                  "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "hed", "hell", "hes",
                  "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "its", "itself", "lets", "me",
                  "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should", "so",
                  "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under",
                  "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "would",
                  "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"]

    # Open file and read lines
    with open(filename_str) as f:
        lines = f.read().splitlines()

    # Add non-stop words to dictionary
    word_id = {}
    for line in lines:
        line = re.sub(r'[^a-zA-Z\s]', "", line)
        words = line.split()
        words = words[2:]
        for word in words:
            word = word.lower()
            if word not in stop_words:
                word_id[word] = 0

    # Assign id's to words in dictionary
    id = 0
    for word in word_id:
        word_id[word] = id
        id += 1

if __name__ == "__main__":
    preprocess_input("test-input.txt")
