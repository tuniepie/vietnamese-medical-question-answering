### Table of Contents
1. [Introduction](#introduction)
2. [Packages/Pre-trained models installation](#installation)
3. [Project baseline](#baseline)

# <a name="introduction"></a> Kalapa challenge: Vietnamese Medical Question Answering 
Competition teams will build a Vietnamese language model capable of answering multiple-choice questions (with one or more correct answers) in the medical field, based on the corpus provided by the organizers (more information in [here](https://challenge.kalapa.vn/portal/vietnamese-medical-question-answering/overview)).

# <a name="installation"></a> Installation
## BeautifulSoup
A library package for processing HTML documents
```
pip install beautifulsoup4 lxml html5lib
```

## Faiss (Facebook AI Similarity Search)
[Faiss](https://faiss.ai/index.html#) default installation for cpu
```
conda install -c pytorch faiss-cpu
```
The `faiss-gpu` package provides CUDA-enabled indices:
```
conda install -c pytorch faiss-gpu
```

## BM25
Quick installation for [BM25](https://github.com/dorianbrown/rank_bm25)
```
pip install rank_bm25 
```

Example usage for `BM25`
```python
from rank_bm25 import BM25Okapi

# Tokenize corpus and initialize model BM25
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Tokenize query
tokenized_query = query.split(" ")

# Retrieve the best documents
bm25.get_top_n(tokenized_query, corpus, n=1)
```

Otherwise, class BM25 can be built manually (more details in [here](https://ndquy.github.io/posts/okapi-bm-25-tim-kiem-tieng-viet/))

## Sentence-Transformers

Quick installation for [Sentence-Transformers](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder)
```
pip install -U sentence-transformers==2.2.2
```

Manually download model cache for `Sentence-Transformers` in case of failed connection to the download directory
```
mkdir cache
cd cache
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder
```

Example usage for `Sentence-Transformers`
```python
from sentence_transformers import SentenceTransformer

# In case running in a jupyter notebook
from huggingface_hub import notebook_login
notebook_login()

model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', cache_folder="./cache")
embeddings = model.encode(sentences)
```

## VNCoreNLP

`Java 1.8+` (Prerequisite) installation for anaconda environment without using sudo
```
conda install -c conda-forge openjdk=11
```

Quick installation for [VNCoreNLP](https://github.com/vncorenlp/VnCoreNLP/tree/master)
```
pip install py_vncorenlp
```
```python
import py_vncorenlp
py_vncorenlp.download_model(save_dir='./cache')
```

Manually download model cache for `VNCoreNLP` in case of failed connection to the download directory
```
git clone https://github.com/vncorenlp/VnCoreNLP.git
mkdir cache
cp -r ./VnCoreNLP/models ./cache
cp ./VnCoreNLP/VnCoreNLP-1.2.jar ./cache
```

Example usage for `VNCoreNLP`
```python
import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./cache')
rdrsegmenter.word_segment(text)
```

# <a name="baseline"></a> Baseline
- Preprocessing corpus dataset by splitting the documents into different sub_docs based on their content with format {title_content_subcontent}.
- Using TF-IDF method (linear kernel as the metrics) for preprocessing dataset to acquire the documents with the most similarity to the queries.
- Using BM25 to take out the top k sentences in the documents with the most similarity to the queries.
- Using cosine similarity metrics to retrieve the final sentence with highest similarity score and the same for the options to get the final answers.
