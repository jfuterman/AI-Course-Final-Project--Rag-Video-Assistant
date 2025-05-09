import os
import gradio as gr
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import json
import subprocess
import imageio_ffmpeg

# Constants
model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    url="https://0d960223-8541-4d94-abc9-ddb490e68e6d.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uDVtAawgI5_xJ6xuHl7L4HqICEDrLxjgFJy7ZCnHbC4",
)

# Use absolute paths
PROJECT_ROOT = "/Users/jessefuterman/rag_video_assistant"
collection_name = "video_chunks"
RAW_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw")
CLIP_FOLDER = os.path.join(PROJECT_ROOT, "demo_clips")

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw folder path: {RAW_FOLDER}")
print(f"Does raw folder exist? {os.path.exists(RAW_FOLDER)}")

# Create output directory
os.makedirs(CLIP_FOLDER, exist_ok=True)
print(f"Clip folder created at: {CLIP_FOLDER}")

# Find all MP4 files
all_mp4_files = []
if os.path.exists(RAW_FOLDER):
    for file in os.listdir(RAW_FOLDER):
        if file.endswith('.mp4'):
            all_mp4_files.append(os.path.join(RAW_FOLDER, file))
    print(f"Found {len(all_mp4_files)} MP4 files in {RAW_FOLDER}")
    for file in all_mp4_files:
        print(f"  - {file}")

# Retrieval
def retrieve_chunks(query, top_k=5):
    embedding = model.encode(query).tolist()
    print(f"[INFO] Embedding vector created for query: '{query}'")
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k,
    )
    print("[INFO] Retrieved chunks:")
    for hit in results:
        print(f"  → {hit.payload['video_id']} | {hit.payload['start_time_ms']}–{hit.payload['end_time_ms']}")
    return results


def format_prompt(query, chunks):
    context = ""
    for idx, hit in enumerate(chunks):
        context += f"Chunk {idx+1} (Video ID: {hit.payload['video_id']}, Start: {hit.payload['start_time_ms']/1000:.2f}s)\n"
        context += f"{hit.payload['combined_text']}\n\n"

    return f"""
You are an educational assistant that provides answers based only on the provided video lecture content.

Student Question:
{query}

Relevant Video Chunks:
{context}

Your Answer:
If the provided chunks don't contain sufficient information about the question topic, please state this clearly. Otherwise, provide a comprehensive answer based only on the information in the chunks.
"""

# LLM generation (Ollama)
def generate_streaming_response(prompt):
    print("[INFO] Sending prompt to Ollama:")
    print(prompt)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:latest",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data.get("response", "")

# Clip extraction using direct ffmpeg
def extract_video_clips(chunks):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Using ffmpeg from: {ffmpeg_path}")
    
    clip_paths = []
    for idx, chunk in enumerate(chunks):
        video_id = chunk.payload['video_id']
        start = chunk.payload['start_time_ms'] / 1000
        end = chunk.payload['end_time_ms'] / 1000
        
        # If start and end are the same or very close, add a small duration
        if abs(end - start) < 0.1:
            end = start + 3.0  # Add 3 seconds to show some context
        
        # Find the matching file
        matching_files = [f for f in os.listdir(RAW_FOLDER) 
                         if f.endswith('.mp4') and video_id in f]
        if not matching_files:
            print(f"⚠️ Video file not found for ID: {video_id}")
            continue
            
        input_path = os.path.join(RAW_FOLDER, matching_files[0])
        print(f"Found matching file for {video_id}: {input_path}")
        output_path = os.path.join(CLIP_FOLDER, f"{video_id}_clip{idx}.mp4")
        
        try:
            print(f"[INFO] Extracting: {video_id} from {start}s to {end}s")
            
            # Use subprocess directly with proper error handling
            import subprocess
            cmd = [
                ffmpeg_path,
                '-i', input_path,
                '-ss', str(start),
                '-to', str(end),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            print(f"Running command: {' '.join(cmd)}")
            
            # Run ffmpeg command and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                clip_paths.append(output_path)
                print(f"Successfully extracted clip to: {output_path}")
            else:
                print(f"ffmpeg error: {stderr.decode()}")
                
        except Exception as e:
            print(f"Failed to extract {video_id}: {e}")
            continue

    return clip_paths

# Chat handler
def gradio_chatbot(message_history):
    if not message_history or not isinstance(message_history, list):
        print("[WARN] Invalid message history format.")
        return [{"role": "assistant", "content": "Invalid chat history format."}], []

    last_message = message_history[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "user" or not last_message.get("content", "").strip():
        print("[WARN] No valid user input.")
        message_history.append({"role": "assistant", "content": "Please enter a valid question."})
        return message_history, []

    query = last_message["content"]
    try:
        # Add interim message to show processing
        message_history.append({"role": "assistant", "content": "Searching for relevant video chunks..."})
        
        chunks = retrieve_chunks(query)
        prompt = format_prompt(query, chunks)
        response_gen = generate_streaming_response(prompt)

        output = ""
        for chunk in response_gen:
            output += chunk

        try:
            video_paths = extract_video_clips(chunks)
            if not video_paths:
                output += "\n\n*Note: Relevant video clips could not be extracted. The text answer is still provided based on the content of the videos.*"
        except Exception as e:
            print(f"Error in video extraction: {e}")
            video_paths = []
            output += f"\n\n*Note: Error extracting video clips, but the text response is still based on the relevant video content.*"
            
        # Replace the interim message with the final response
        message_history[-1] = {"role": "assistant", "content": output}
        return message_history, video_paths

    except Exception as e:
        print(f"[ERROR] Exception in gradio_chatbot(): {e}")
        message_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
        return message_history, []

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Video and Text Display Based on Question")
    
    with gr.Group():
        question_input = gr.Textbox(label="Ask a question", placeholder="What is logistic regression?")
    
    with gr.Group():
        with gr.Tab("Video"):
            video_output = gr.Video(label="Video Answer")
        
        answer_text = gr.Markdown()

    def handle_question(question):
        try:
            # Get chunks from vector DB
            chunks = retrieve_chunks(question)
            
            # Generate text response
            prompt = format_prompt(question, chunks)
            response_gen = generate_streaming_response(prompt)
            text_answer = ""
            for chunk in response_gen:
                text_answer += chunk
            
            # Extract video clips
            video_paths = extract_video_clips(chunks)
            
            if video_paths:
                return video_paths[0], text_answer
            else:
                return None, text_answer + "\n\n*No relevant video clip found.*"
                
        except Exception as e:
            return None, f"Error: {str(e)}"

    question_input.submit(handle_question, [question_input], [video_output, answer_text])

demo.launch()