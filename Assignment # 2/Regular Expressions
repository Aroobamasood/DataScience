# 29 September 2023
# CSC461 – Assignment2 – Regular Expressions
# AROOBA MASOOD
# FA20-BSE-092
"""
The following code is using python re library to extract some things from the txt file. The code reads text from a file and extracts the following:
1. It uses a regular expression to identify words and stores them in the list_of_words variable.
2. It extracts a list of all words that start with a capital letter using another regular expression pattern.
3. It extracts a list of all words with a length of exactly 5 characters.
4. It finds and extracts a list of words that are enclosed within double quotes.
5. It extracts all vowels, both uppercase and lowercase from the text.
6. It extracts a list of 3-letter words that end with the letter 'e' using a regular expression.
7. It identifies and extracts a list of words that both start and end with the letter 'b' using a regular expression.
8. IT removes all punctuation marks from the text.
9. It replaces words ending in 'n't' with their full form 'not' using the re.sub() function.
10. It replaces all new line characters with a single space to format the text.
"""

import re
with open('example-text.txt', 'r') as file:
    text = file.read()

    # Extract list of all words.
    list_of_words = re.findall(r"\b\w+(?:'\w+)?\b", text)
    print("\033[1mList of all words are:\033[0m")
    print(list_of_words)

    # Extract list of all words starting with a capital letter.
    Capital_words = re.findall(r"\b[A-Z].*?\b", text)
    print("\n\033[1mList of all words starting with Capital letter are:\033[0m")
    print(Capital_words)

    # Extract list of all words starting with a capital letter.
    length_of_five = re.findall(r"\b\w{5}\b", text)
    print("\n\033[1mList of all words of length 5 are:\033[0m")
    print(length_of_five)

    # Extract list of all words inside double quotes.
    quoted_words = re.findall(r'"([^"]*)"', text)
    print("\n\033[1mList of all words inside double quotes are:\033[0m")
    print(quoted_words)

    # Extract list of all vowels.
    vowels = re.findall(r'[AEIOUaeiou]', text)
    print("\n\033[1mList of all vowels are:\033[0m")
    print(vowels)

    # Extract list of 3 letter words ending with letter ‘e’.
    length_of_three = re.findall(r'\b\w{2}e\b', text)
    print("\n\033[1mList of all vowels are:\033[0m")
    print(length_of_three)

    # Extract list of all words starting and ending with letter ‘b’.
    start_b = re.findall(r'\b[bB]\w*?b\b', text)
    print("\n\033[1mList of all words starting and ending with letter ‘b’ are:\033[0m")
    print(start_b)

    # Remove all the punctuation marks from the text.
    punctuation = re.sub(r'[^\w\s]', '',  text)
    print("\n\033[1mRemoving all the punctuation marks from the text:\033[0m")
    print(punctuation)

    # Replace all words ending ‘n't’ to their full form ‘not’.
    replace_word = re.sub("n't", ' not', text)
    print("\n\033[1mReplacing all words ending ‘n't’ to their full form ‘not’:\033[0m")
    print(replace_word)

    # Replace all the new lines with a single space.
    replace_lines = re.sub('\n+', ' ', text)
    print("\n\033[1mReplacing all new lines with a single space:\033[0m")
    print(replace_lines)
