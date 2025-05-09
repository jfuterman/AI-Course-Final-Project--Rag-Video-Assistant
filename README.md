# AI-Course-Final-Project---RAG-Video-Assistant

Key Features

- Semantic Search: Uses `all-MiniLM-L6-v2` to find the most relevant video chunks.
- Clip-Based RAG: Retrieves exact moments from videos to generate responses.
- LLM Integration: Streams answers using Ollama and Mistral.
- Video Highlights: Dynamically extracts and displays matching clips.
- Transcript Support: Displays relevant transcript text alongside each answer.
- Gradio UI: Clean and interactive front-end for user queries and responses.

How It Works

1. Frame Extraction & Subtitles:
   - Subtitle text is aligned with video frames and stored as JSON.

2. Chunking:
   - Frames are grouped using hybrid chunking based on time and semantic similarity.

3. Featurization:
   - Each chunk is embedded using SentenceTransformers and stored in a Qdrant vector database.

4. Query Flow:
   - A user asks a question.
   - The system retrieves the top matching chunks.
   - A prompt is formatted and sent to an LLM via Ollama.
   - The final answer is streamed to the UI with the associated video clip and transcript.


Installation

1. Clone the Repository

2. Create a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```
pip install --no-cache-dir -r requirements.txt
```
> If `moviepy` gives issues, use `pip install --no-cache-dir moviepy==1.0.3`


Running the App

Ensure Ollama is running locally with `mistral` pulled:
```
ollama run mistral
```
Then launch the app:
```
python app.py
```

Visit [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.


File Structure

File Structure
```
rag_video_assistant/
├── app.py                          # Main Gradio interface
├── test_files.py                   # Test script (optional)
├── requirements.txt                # Dependencies
├── README.md                       # Project instructions

├── 01_data_collection.ipynb        # Subtitle + frame extraction
├── 02_chunking.ipynb               # Chunk generation logic
├── 03_featurization.ipynb          # Embedding + Qdrant upload
├── 04_retrival.ipynb               # Retrieval logic testing
├── 05_gradio_ollama_app.ipynb      # App UI logic
├── 06_demonstration.ipynb          # Final sample run-through

├── data/
│   ├── raw/                        # Source lecture videos (.mp4)
│   └── processed/                 # Frames, chunks, and metadata (JSON)

├── demo_clips/                     # Highlighted video excerpts
```


Environment Setup

Make sure your `.env` or code contains access to:

- Qdrant API key and URL
- Ollama running locally
- Proper video file paths (`data/raw/`)
- Transcripts with matching `video_id`

Example Questions to Try

1. “How do ResNets solve the vanishing gradient problem?”
2. “What is the purpose of temporal difference learning?”
3. “Compare CNNs and fully connected networks.”
4. “What does Bayes filter do in a reinforcement learning context?”


