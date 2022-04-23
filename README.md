# Document-summarization-and-key-inforamtion-extraction
CS521 Statistical natural language processing language project

https://docs.google.com/presentation/d/e/2PACX-1vRpihznTgaP547SAc97yclb8gSzI3tTuj2ZRakwCxQxAuIqb5P06wgYiqvGfmbJXD3lAPxV88JOKM9g/pub?start=false&loop=false&delayms=3000

This code is for the implementation of document summarization with HuggingFace Transformers - Pegasus, T5, distibart
# Data Preparation
Xsum dataset, which has BBC news articles text and their summaries, can be loaded using below code

dataset = load_dataset("xsum")

or
data can be downloaded from https://huggingface.co/datasets/xsum

For fine tuning HuggingFace transformers 1k data rows are considered 

# Modeling
### Pre-trained tranformer models
An encoder-decoder models are pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format. T5 works well on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task, e.g., for translation: translate English to German: …, for summarization: summarize: ….

These models have millions of parameters and are trained on huge data.

### Fine tune with xsum data
models are fine tuned with xsum dataset to perform downstreamtasks, here summarization task

# Long Document Summarization
For the very long documents, sentences are convereted to chunks such that the number of tokens in each chunk doesn't exceed 1024. Each chunk is summarized with the model. I summarized 2 long PDFs and extracted key information from the summary and then performed Question-Answering task on the summary. These two PDFs can be found in the Data folder. Summary results of these two long documents are in the output folder.

# Model Evaluation
Both pre-trained and fine-tune Models are evaluated using Rouge score(Rouge1, Rouge2, RougeL) and cosine similarity.
