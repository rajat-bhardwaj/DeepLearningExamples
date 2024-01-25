# RAG chatbot with Falcon 7b instruct

This is an example of using RAG with `Falcon-7b-instruct` available on Hugging Face. We have used the `tiny strange textbooks` dataset to generate documents for "domain" specific retrieval.

The [model](https://huggingface.co/tiiuae/falcon-7b-instruct) and [dataset](https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks) are both available on HF.

FAISS: A vector database for storing the embeddings and computing similarity.

ChainLit provided the necessary chatUI for running the chatbot.


## Setup
- Start by installing the dependencies in `requirements.txt`.

- Set up the `.env` file with your HF token.

`HUGGINGFACEHUB_API_TOKEN=<TOKEN>`

- run app.py via `chainlit run app.py -w`. It will launch the chatbot app.

## ToDo

This example only uses one file to populate the vector database. To run it on the entire dataset (requires more resources), comment line 101 `data_files="data_part_0.parquet"` in `falcom7b_RAG_chainlit.py`.

The `train` parameter controls the embedding generation and FAISS datastore creation. Set it to `true` to run the example the first time. It will create and store the vectors in a file. For subsequent runs, the vector store will be loaded from the file. (it is an expensive operation even with GPU)