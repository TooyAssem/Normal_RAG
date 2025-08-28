import os
import time
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- User Instructions ---
# 1. Set your Groq API key as an environment variable named "GROQ_API_KEY".
#    You can get a key from https://console.groq.com/keys
#    Example (in your terminal): export GROQ_API_KEY='your_key_here'
# 2. Replace the placeholder content in the `content` variable below with the text you want to summarize.
# ---

def get_completion(prompt, temperature=0.7, model="mixtral-8x7b-32768"):
    """Generate a completion using the Groq API."""
    try:
        # Initialize Groq client inside the function to use the environment variable
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable.")

        client = Groq(api_key=groq_api_key)

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during get_completion: {e}")
        return None

def retrieve_with_sentence_window(qdrant_client, model, content, query, k=3, window_size=1):
    """Retrieve relevant sentences and their surrounding context (window)."""
    # 1. Split content into sentences
    sentence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=[". ", ".\n", "!", "?", "\n"],
        keep_separator=True
    )
    sentences = sentence_splitter.split_text(content)

    # 2. Create a new collection for sentences
    sentence_collection_name = "sentence_collection"
    if qdrant_client.collection_exists(collection_name=sentence_collection_name):
        qdrant_client.delete_collection(collection_name=sentence_collection_name)

    qdrant_client.create_collection(
        collection_name=sentence_collection_name,
        vectors_config=models.VectorParams(size=model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
    )

    # 3. Generate embeddings and upload to Qdrant
    embeddings = model.encode(sentences)
    qdrant_client.upload_points(
        collection_name=sentence_collection_name,
        points=[
            models.PointStruct(id=i, vector=embedding.tolist(), payload={"text": sentence, "original_index": i})
            for i, (sentence, embedding) in enumerate(zip(sentences, embeddings))
        ]
    )

    # 4. Retrieve the most relevant sentences
    query_embedding = model.encode([query])[0]
    results = qdrant_client.query_points(
        collection_name=sentence_collection_name,
        query=query_embedding.tolist(),
        limit=k
    ).points

    # 5. Get the windows of sentences
    retrieved_indices = sorted([result.payload["original_index"] for result in results])

    context_sentences = []
    added_indices = set()

    for index in retrieved_indices:
        start = max(0, index - window_size)
        end = min(len(sentences) - 1, index + window_size)

        for i in range(start, end + 1):
            if i not in added_indices:
                context_sentences.append(sentences[i])
                added_indices.add(i)

    # 6. Concatenate sentences to form the context
    return "".join(context_sentences)

if __name__ == '__main__':
    # --- Content to be summarized ---
    # Please replace this with your own content.
    content = """
    Increasing Notch signaling antagonizes PRC2-mediated silencing to promote reprograming of germ cells into neurons.
    Cell-fate decisions are controlled, on the one hand, by intercellular signaling and, on the other hand, by intrinsic mechanisms such as epigenetic chromatin modifications.
    The Notch signaling pathway is a highly conserved and widespread signaling mechanism, which has been implicated in key cell-fate decisions such as the decision between proliferation and differentiation.
    Notch signaling has also been implicated in cellular reprograming.
    """

    start_time = time.time()

    # Initialize SentenceTransformer model
    print("Initializing SentenceTransformer model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    print("Model initialized.")

    # Initialize Qdrant client (in-memory)
    print("Initializing Qdrant client...")
    qdrant_client = QdrantClient(":memory:")
    print("Qdrant client initialized.")

    # Retrieve context using the new sentence window method
    print("Retrieving context with sentence window...")
    retrieval_query = "What is the role of Notch signaling in reprograming?"
    retrieved_context = retrieve_with_sentence_window(
        qdrant_client,
        model,
        content,
        retrieval_query,
        k=1,
        window_size=1
    )
    print("Context retrieved.")

    # Build summarization prompt
    summarization_prompt = f"""Based on the following context, please provide a summary:

    **Context:**
    {retrieved_context}

    Create a comprehensive summary of the provided text."""

    # Generate summary
    print("Generating summary...")
    summary = get_completion(summarization_prompt)
    print("Summary generated.")

    print("\n--- Summary (Context Engineering) ---")
    print(summary)
    print("---------------------------------------")

    print(f"\nTotal execution time: {(time.time() - start_time):.2f} seconds")
