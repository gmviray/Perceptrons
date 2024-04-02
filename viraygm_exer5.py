# VIRAY, Geraldine Marie M.
# CMSC 170 X5L
# Exercise 05: Naïve Bayes and Laplace Smoothing

import os
import re
from decimal import *

# function to loop all files in the directory
def file_loop(directory_path):
    files = []
    file_count = 0

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            files.extend(read_file(file_path))
            file_count += 1

    return files, file_count

# function to read contents of a file
def read_file(file_path):
    words = []

    with open(file_path, "r", encoding='latin-1') as file:
        for line in file:
            words.extend(line.split())

    return words

# function that removes all alphanumeric characters
def remove_nonalphanumeric_lowercase(string_list):
    return [re.sub(r'[\W_]+', '', s).lower() for s in string_list if re.sub(r'[\W_]+', '', s)]

# function for finding any duplicates in the bag of words
def find_duplicates(bow, key):
    for i in bow:
        if bow[i]["Word"] == key:
            return i
    return -1

# function for generating bag of words
# if_checked is a flag to check if existing words in the bow should be considered
def bag_of_words(word_list, if_checked):
    # clean the words in word_list
    cleaned_text = remove_nonalphanumeric_lowercase(word_list)

    # empty dictionary, to store the bag of words
    bow = {}

    # for counting
    bow_index = 0

    for word in cleaned_text:
        # find duplicates of word in bow
        index = find_duplicates(bow, word)

        if if_checked and index != -1:
            bow[index]["Frequency"] += 1
        else:
            bow[bow_index] = {"Word": word, "Frequency": bow.get(index, {"Frequency": 0})["Frequency"] + 1}
        
        bow_index += 1

    return bow_index, bow

# calculate the size of the combined dictionary
def calculate_dict_size(spam, ham):
    # unique elements of spam and ham
    combined_spam_ham = list(set(spam).union(ham))
    return len(combined_spam_ham), combined_spam_ham

# calculates new words that are not in spam and ham combined
def count_new_words(combined_text, combined_spam_ham):
    new_word_count = 0
    for word in combined_text.split():
        if word not in combined_spam_ham:
            new_word_count += 1
    return new_word_count

# function for counting the frequency of a word in a bag of words
def count_word_frequency(bow, text):
    return next((entry['Frequency'] for entry in bow.values() if entry['Word'] == text), 0)

# total probability of a message being classified as spam or ham
def calculate_total_probability(combined_text, bow, k, count, dict_size, new_word_count, p):
    total = 1
    for text in combined_text.split():
        # counts word frequency in bow
        count_bow = count_word_frequency(bow, text)
        subtotal = Decimal(count_bow + k) / Decimal(count + k * (dict_size + new_word_count))
        total *= subtotal
    new_total = total * Decimal(p)
    return new_total.ln()

# calculates the probability of a message being spam
def spam_probability(total_spam, total_ham):
    return Decimal(total_spam.exp()) / (Decimal(total_spam.exp()) + Decimal(total_ham.exp()))

# function to write in classify.out file
def write_output(spam_ham, ham_bow, ham_word_count, spam_bow, spam_word_count):
    index = 1

    with open('classify.out', 'w+') as file:
        for line in spam_ham:
            is_spam = "SPAM" if line > 0.5 else "HAM"
            file.write(f'{str(index).zfill(3)} {is_spam} {str(line)}\n')
            index += 1

        file.write('\nHAM\n')
        file.write(f'Dictionary Size: {len(ham_bow)}\n')
        file.write(f'Total Number of Words: {ham_word_count}\n')

        file.write('\nSPAM\n')
        file.write(f'Dictionary Size: {len(spam_bow)}\n')
        file.write(f'Total Number of Words: {spam_word_count}\n')

    print("\nclassify.out successfully generated.")

ham_directory_path = r'./data/data02/ham'

# read ham files and count them
ham_files, ham_file_count = file_loop(ham_directory_path)

# get bag of words and its word count
ham_word_count, ham_bow = bag_of_words(ham_files, True)

spam_directory_path = r'./data/data02/spam'

# read spam files and count them
spam_files, spam_file_count = file_loop(spam_directory_path)

# get bag of words and its word count
spam_word_count, spam_bow = bag_of_words(spam_files, True)

spam_size = len(spam_bow)
ham_size = len(ham_bow)

# for printing
print("HAM")
print("Dictionary Size: " + str(ham_size))
print("Total Number of Words: " + str(ham_word_count))
print("\n")
print("SPAM")
print("Dictionary Size: " + str(spam_size))
print("Total Number of Words: " + str(spam_word_count))

# get k
smoothing_factor = int(input("Enter a smoothing factor: "))

# calculate probabilities of a message being a spam or ham using Laplace smoothing in a naive bayes classifier
p_spam = (spam_file_count + smoothing_factor) / ((spam_file_count + ham_file_count) + 2 * smoothing_factor)
p_ham = (ham_file_count + smoothing_factor) / ((spam_file_count + ham_file_count) + 2 * smoothing_factor)

# getting the size and content of unique words of spam and ham
combined_size, combined_spam_ham = calculate_dict_size(spam_files, ham_files)

# store probabilities
spam_ham = []
classify_directory_path = r'./data/data02/classify'

# goes through all files in the directory
for filename in os.listdir(classify_directory_path):
    file_path = os.path.join(classify_directory_path, filename)

    # checks if valid file
    if os.path.isfile(file_path):
        # reads file content
        text_file = read_file(file_path)

        # generate bag of words
        word_count, file_bow = bag_of_words(text_file, False)

        # combines words
        combined_text = ' '.join(entry['Word'] for entry in file_bow.values())

        # count new words
        new_word_count = count_new_words(combined_text, combined_spam_ham)

        # calculates the total probability that the current file is spam
        total_spam = calculate_total_probability(combined_text, spam_bow, smoothing_factor, spam_word_count, combined_size, new_word_count, p_spam)
        
        # calculates the total probability that the current file is ham
        total_ham = calculate_total_probability(combined_text, ham_bow, smoothing_factor, ham_word_count, combined_size, new_word_count, p_ham)

        # computes the final probability that the file is spam or ham based on its properties
        p_spam_text = spam_probability(total_spam, total_ham)
        spam_ham.append(p_spam_text)

write_output(spam_ham, ham_bow, ham_word_count, spam_bow, spam_word_count)
