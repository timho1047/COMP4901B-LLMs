import argparse
import re
from numpy.ma import masked
import requests
import json
from utils import  read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string
from bs4 import BeautifulSoup


def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)


def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    PATTERN_AND_MASK = [(r'\b\d{3}-\d{2}-\d{4}\b', "XXX-XX-XXXX"), (r'\+1\d{10}', "X" * 11)]
    for pattern, mask in PATTERN_AND_MASK:
        text = re.sub(pattern, mask, text)
    return text
    

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """

    ALNUM_COUNT_THRESHOLD = 101
    
    cleaned = []
    for para in text.split("\n"):
        
        # Alphanumeric count gate
        if re.search(rf'\w{{{ALNUM_COUNT_THRESHOLD},}}', para):
            continue # remove paragraphs that have too many consecutive alphanumeric characters
        
        if set(para) & set(string.punctuation) == set():
            continue # remove paragraphs that have no punctuation
        
        cleaned.append(para)
    return "\n".join(cleaned)


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    ALNUM_RATIO_THRESHOLD = 0.8
    char_set = set(text)
    
    # Bad word gate
    bad_words = retrieve_bad_words()
    if any(word.lower() in bad_words for word in text.split()):
        return False
    
    # Punctuation gate, has at least one punctuation character
    punctuation = set(string.punctuation)
    if punctuation & char_set == set():
        return False

    # Whitespace gate, has non-whitespace characters
    whitespace = set(string.whitespace)
    if char_set - (whitespace | punctuation) == set():
        return False
    
    # Alphanumeric gate, has at least 80% alphanumeric, punctuation, or whitespace characters
    alnum_ratio = sum(c.isalnum() or c in (whitespace | punctuation) for c in text) / len(text)
    if alnum_ratio < ALNUM_RATIO_THRESHOLD:
        return False
    
    return True


def is_english_text(text: str) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """
    ENGLISH_CHAR_RATIO_THRESHOLD = 0.99
    all_alphas = [c for c in text if c.isalpha()] # only consider alphabetic characters
    english_alphas = [c for c in all_alphas if re.match(r'[a-zA-Z]', c) is not None] # only consider English alphabetic characters
    english_ratio = len(english_alphas) / len(all_alphas) if all_alphas else 0
    return english_ratio > ENGLISH_CHAR_RATIO_THRESHOLD
    

def deduplicate_texts(texts: list[str]) -> list[str]:
    """Deduplicates text by removing duplicate sentences.
    Args:
        texts (list[str]): List of text strings to deduplicate.
    Returns:
        list[str]: Deduplicated list of texts. Implemented a simple Jaccard similarity based deduplication.
    """
    SIMILARITY_THRESHOLD = 0.4
    
    def jaccard_similarity(set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))
    
    text_set_pairs = [(text, set(text.lower().split())) for text in texts] # tokenize by words
    deduplicated = []
    for text, text_set in text_set_pairs:
        for _, unique_text_set in deduplicated:
            sim = jaccard_similarity(text_set, unique_text_set)
            if sim > SIMILARITY_THRESHOLD:
                break
        else:
            deduplicated.append((text, text_set)) # keep the original text and its set
    return [text for text, _ in deduplicated]


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = '', help = 'Specify the path for your warc file.')
    parser.add_argument('--dfname', type = str,  default = '', help = 'Specify the path where you stored topic_dataset.json')
    parser.add_argument('--num_records', type = int,  default=30, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    parser.add_argument('--output', type = str,  default='cleaned_documents.txt', help = 'Output file for cleaned text documents')
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, 'w', encoding='utf-8') as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                # print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = cleaned_nopii_text.replace('\n', ' ').replace('\r', ' ').strip()
                    output_file.write(single_line_text + '\n')
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, 'r') as f:
            raw_texts = json.load(f)
        raw_texts = [item['text'] for item in raw_texts['data']]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed.")
    else:
        print("Usage: python homework.py --fname data.warc")