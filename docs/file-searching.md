# File Searching
The implementation is fairly simple. It's intentionally not async to keep things simple, but one could imagine that at scale you would want to make part of the processing async. The idea for what happens with file uploads/searches is explained in the diagram below, but essentially:
1. Files are uploaded and chunked into 500-token-sized chunks and embeddings are created.
2. When the file tools are utilized, we convert the query into an embedding and perform an embedding search to find chunks which are semantically close to the data we are looking for. For images we generate a description of the image with an LLM and then generate an embedding for that description. This way we can allow for easy contextual searches of images as well.

```mermaid
flowchart TD
    A[File Uploaded] --> B[File Type Check]
    B -->|Image| C[Generate Image Description via LLM]
    B -->|Text / PDF / DOCX| E[Extract Text]
    C --> F[Single Chunk from Description]
    E --> F2[Split into 500-Token Chunks]
    F --> G[Create Embedding per Chunk]
    F2 --> G
    G --> H[Save File Information + Chunks to DB]
```
