import re


def tokenize(text):

    text = text.lower()

    tokens = re.findall(r"\w+|[^\w\s]", text)

    return tokens