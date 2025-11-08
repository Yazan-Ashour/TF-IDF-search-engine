# 🔍 TF-IDF Document Ranking System (VSM + Cosine Similarity)

This Python application ranks a collection of `.txt` documents based on their **relevance to a query** using the **Vector Space Model (VSM)** with **TF-IDF weighting** and **cosine similarity**.  
It also includes **query expansion** using **WordNet**, **lemmatization**, and configurable preprocessing options.

---

##  Overview

Given:
- A **folder path** containing multiple text files.
- A **query string**.

The program:
1. Preprocesses each document (tokenization, lowercasing, lemmatization).
2. Expands the user query using WordNet synonyms.
3. Converts all documents to **TF-IDF vectors** using `CountVectorizer` and `TfidfTransformer`.
4. Computes the **cosine similarity** between the query vector and all document vectors.
5. Displays the **Top-5 most relevant documents** with:
   - Their file paths
   - Cosine similarity scores
   - A short content preview with query terms highlighted

---

##  Features

- **TF-IDF Vectorization** using `scikit-learn`
- **Cosine Similarity** for document ranking
- **Query Expansion** using WordNet (synonyms for nouns & verbs)
- **Lemmatization** using NLTK’s WordNet lemmatizer
- **Stopword Removal** and `CountVectorizer` parameters:
  - `lowercase=True`
  - `stop_words='english'`
  - `min_df=1`
  - `max_features=5000`
  - `ngram_range=(1, 2)`
- **Colored console output** with `colorama`
- **Performance reporting** (execution time, vocabulary size, etc.)

---

##  Dependencies

Make sure you have the following installed:

```bash
pip install numpy scikit-learn nltk colorama
```

Additionally, download required NLTK data files:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

##  How to Run

### **1. Prepare your folder**
Create a folder containing multiple `.txt` files — each file represents one document.

Example:
```
documents/
├── doc1.txt
├── doc2.txt
├── doc3.txt
```

### **2. Run the program**
Use the terminal or command prompt:

```bash
python TFIDF.py <folder_path> <query>
```

Example:
```bash
python TFIDF.py ./documents "machine learning applications"
```

### **3. Sample Output**

```
=================================================================
TF-IDF DOCUMENT RANKING SYSTEM
=================================================================
Folder Path: ./documents
Original Query: machine learning applications
Expanded Query: applications learning machine

Loaded 10 documents.
Vocabulary Size: 487 terms

Top 5 Relevant Documents:

RANK  SIMILARITY     DOCUMENT PATH
-------------------------------------------------------------
1     0.781235       ./documents/doc3.txt
2     0.624513       ./documents/doc1.txt
3     0.583492       ./documents/doc5.txt
4     0.417284       ./documents/doc4.txt
5     0.376918       ./documents/doc2.txt
-------------------------------------------------------------
Most Relevant Document: doc3.txt
Average Top-5 Similarity: 0.556688
Execution Time: 1.48 seconds
=================================================================

DETAILED DOCUMENT PREVIEWS
=================================================================
[1] doc3.txt
-----------------------------------------------------------------
...the [machine] [learning] algorithm improved...
-----------------------------------------------------------------
```

---

##  File Structure

```
TFIDF.py               # Main program
README.md              # Documentation
documents/             # Folder containing .txt files
```

---

##  Notes on Parameters

| Parameter | Description | Impact |
|------------|-------------|--------|
| `lowercase` | Converts all text to lowercase | Ensures case-insensitive matching |
| `stop_words='english'` | Removes common English words | Improves relevance by ignoring filler words |
| `min_df=1` | Ignores terms that appear in fewer than 1 document | Keeps all terms |
| `max_features=5000` | Limits vocabulary size | Prevents overfitting and speeds up computation |
| `ngram_range=(1,2)` | Uses unigrams and bigrams | Captures simple word combinations |

---

##  Example Use Case

You can use this project for:
- Document search engines
- Text similarity and ranking experiments
- Exploring the effect of TF-IDF and cosine similarity in practice

---

##  Author

**Developed by:** Yazan Ashour 
**Language:** Python 3  
**Libraries:** `scikit-learn`, `nltk`, `numpy`, `colorama`

