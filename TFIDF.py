import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from colorama import Fore, Style, init

init(autoreset=True)
lemmatizer = WordNetLemmatizer()

# -----------------------------------------
# TEXT PREPROCESSING
# -----------------------------------------
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return " ".join(lemmatized)

# -----------------------------------------
# QUERY EXPANSION USING WORDNET
# -----------------------------------------
def expand_query(query):
    stop_expansion = {"to", "be", "or", "not", "the", "and", "a", "of", "in", "on", "for", "at"}
    expanded_terms = set()

    for word in query.split():
        word = word.lower().strip()
        if word in stop_expansion:
            continue

        synsets = wn.synsets(word)
        for syn in synsets[:2]:
            if syn.pos() not in ['n', 'v']:
                continue
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ').lower()
                if len(lemma_name) < 3 or not lemma_name.isalpha() or lemma_name == word:
                    continue
                expanded_terms.add(lemma_name)

    final_expanded = set(query.split()) | expanded_terms
    return " ".join(sorted(final_expanded))

# -----------------------------------------
# LOAD DOCUMENTS
# -----------------------------------------
def load_documents(folder_path):
    docs = []
    paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            with open(path, "r", encoding="utf-8") as f:
                content = preprocess_text(f.read())
                docs.append(content)
                paths.append(path)
    return docs, paths

# -----------------------------------------
# TF-IDF SEARCH FUNCTION
# -----------------------------------------
def run_search(folder_path, query):
    start_time = time.time()

    print("=" * 65)
    print("TF-IDF DOCUMENT RANKING SYSTEM")
    print("=" * 65)
    print(f"Folder Path: {folder_path}")
    print(f"Original Query: {query}")

    expanded_query = expand_query(query)
    print(f"Expanded Query: {expanded_query}")

    documents, file_paths = load_documents(folder_path)
    if not documents:
        print("No text files found in the provided folder.")
        sys.exit(1)

    print(f"\nLoaded {len(documents)} documents.")

    # TF-IDF Vectorization
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words='english',
        min_df=1,
        max_features=5000,
        ngram_range=(1, 2)
    )

    counts = vectorizer.fit_transform(documents)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counts)

    print(f"Vocabulary Size: {len(vectorizer.get_feature_names_out())} terms\n")

    # Process Query
    query_processed = preprocess_text(expanded_query)
    query_count = vectorizer.transform([query_processed])
    query_tfidf = transformer.transform(query_count)

    similarities = cosine_similarity(query_tfidf, tfidf).flatten()
    top_k = 5
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # RESULTS TABLE
    print("Top 5 Relevant Documents:\n")
    print(f"{'RANK':<6}{'SIMILARITY':<15}DOCUMENT PATH")
    print("-" * 80)

    previews = []  # store previews for final section
    for rank, idx in enumerate(top_indices, start=1):
        if rank == 1:
            color = Fore.GREEN
        elif rank <= 3:
            color = Fore.YELLOW
        else:
            color = Fore.WHITE

        print(color + f"{rank:<6}{similarities[idx]:<15.6f}{file_paths[idx]}" + Style.RESET_ALL)

        with open(file_paths[idx], "r", encoding="utf-8") as f:
            snippet = f.read(300).replace("\n", " ")
        for term in query_processed.split():
            snippet = snippet.replace(term, f"[{term.upper()}]")
        previews.append((rank, file_paths[idx], snippet))

    avg_sim = np.mean(similarities[top_indices])
    print("-" * 80)
    print(f"Most Relevant Document: {os.path.basename(file_paths[top_indices[0]])}")
    print(f"Average Top-5 Similarity: {avg_sim:.6f}")
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print("=" * 65)

    # PREVIEW TABLES SECTION
    print("\nDETAILED DOCUMENT PREVIEWS")
    print("=" * 65)
    for rank, path, snippet in previews:
        title = os.path.basename(path)
        print(f"\n[{rank}] {title}")
        print("-" * 65)
        print(snippet + "...")
        print("-" * 65)


# -----------------------------------------
# MAIN FUNCTION
# -----------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python TFIDF.py <folder_path> <query>")
        sys.exit(1)

    folder_path = sys.argv[1]
    query = " ".join(sys.argv[2:]).strip()

    # Check for empty query
    while not query:
        print(Fore.RED + "\nPLEASE ENTER A QUERY:" + Style.RESET_ALL)
        query = input("> ").strip()

    run_search(folder_path, query)


# -----------------------------------------
if __name__ == "__main__":
    main()