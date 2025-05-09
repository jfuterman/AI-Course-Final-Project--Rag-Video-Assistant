import os
import gradio as gr
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
import glob

# Constants
model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    url="https://0d960223-8541-4d94-abc9-ddb490e68e6d.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uDVtAawgI5_xJ6xuHl7L4HqICEDrLxjgFJy7ZCnHbC4",
)

# Use absolute paths based on the project root
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

# List all MP4 files in the project
def find_all_mp4_files():
    """Search for all mp4 files in the project directory and subdirectories"""
    all_mp4_files = []
    
    # Start with standard data/raw location
    if os.path.exists(RAW_FOLDER):
        mp4_files = glob.glob(os.path.join(RAW_FOLDER, "*.mp4"))
        all_mp4_files.extend(mp4_files)
        print(f"Found {len(mp4_files)} MP4 files in {RAW_FOLDER}")
    
    # Also search the project root
    root_mp4_files = glob.glob(os.path.join(PROJECT_ROOT, "*.mp4"))
    all_mp4_files.extend(root_mp4_files)
    print(f"Found {len(root_mp4_files)} MP4 files in project root")
    
    # Also search any data folder
    data_folder = os.path.join(PROJECT_ROOT, "data")
    if os.path.exists(data_folder):
        data_mp4_files = glob.glob(os.path.join(data_folder, "*.mp4"))
        all_mp4_files.extend(data_mp4_files)
        print(f"Found {len(data_mp4_files)} MP4 files in data folder")
        
    # Print all found MP4 files
    print("\nAll found MP4 files:")
    for file in all_mp4_files:
        print(f"  - {file}")
        
    return all_mp4_files

# Call this at startup to find all MP4 files
ALL_MP4_FILES = find_all_mp4_files()

# --- Retrieval ---
def retrieve_chunks(query, top_k=5):
    embedding = model.encode(query).tolist()
    print(f"[INFO] Embedding vector created for query: '{query}'")
    # Switch back to using search method which we know works (even with deprecation warning)
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
    if not chunks:
        return []

    #   Only extract the top-1 chunk
    chunk = chunks[0]
    video_id = chunk.payload['video_id']
    start = chunk.payload['start_time_ms'] / 1000
    end = chunk.payload['end_time_ms'] / 1000

    # Patch short clips: extend to 2 minutes if possible
    desired_duration = 120
    if end - start < desired_duration:
        end = start + desired_duration


    return f"""
You are an educational assistant that provides answers based only on the provided video lecture content.

Student Question:
{query}

Relevant Video Chunks:
{context}

Your Answer:
If the provided chunks don't contain sufficient information about ResNets or the question topic, please state this clearly. Otherwise, provide a comprehensive answer based only on the information in the chunks.
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

# Improved clip extraction with flexible file matching
def extract_video_clips(chunks):
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Using ffmpeg from: {ffmpeg_path}")

    if not chunks:
        print("No chunks provided for extraction.")
        return []

    # Only extract the most relevant (top-1) chunk
    chunk = chunks[0]
    video_id = chunk.payload['video_id']
    start = chunk.payload['start_time_ms'] / 1000
    end = chunk.payload['end_time_ms'] / 1000

    # Extend short clips to ~2 minutes if possible
    desired_duration = 120
    if end - start < desired_duration:
        end = start + desired_duration

    # Find the matching file
    matching_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith('.mp4') and video_id in f]
    if not matching_files:
        print(f"⚠️ Video file not found for ID: {video_id}")
        return []

    input_path = os.path.join(RAW_FOLDER, matching_files[0])
    output_path = os.path.join(CLIP_FOLDER, f"{video_id}_highlight.mp4")

    try:
        print(f"Extracting {video_id} from {start:.2f}s to {end:.2f}s — Duration: {end - start:.2f}s")

        from moviepy.editor import VideoFileClip

        with VideoFileClip(input_path) as clip:
            duration = clip.duration
            end = min(end, duration)  # Clip cannot exceed actual video length
            subclip = clip.subclip(start, end)
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        print(f"Successfully saved highlight to: {output_path}")
        return [output_path]

    except Exception as e:
        print(f"Failed to extract video: {e}")
        return []


# Chat handler
def gradio_chatbot(message_history):
    if not message_history or not isinstance(message_history, list):
        print("[WARN] Invalid message history format.")
        return [{"role": "assistant", "content": "Invalid chat history format."}], None, "*No transcript found.*"

    last_message = message_history[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "user" or not last_message.get("content", "").strip():
        print("[WARN] No valid user input.")
        message_history.append({"role": "assistant", "content": "Please enter a valid question."})
        return message_history, None, "*No transcript found.*"

    query = last_message["content"]
    try:
        # Add message to show processing
        message_history.append({"role": "assistant", "content": "Searching for relevant video chunks..."})
        
        chunks = retrieve_chunks(query)
        prompt = format_prompt(query, chunks)
        response_gen = generate_streaming_response(prompt)

        output = ""
        for chunk in response_gen:
            output += chunk

        # Extract transcript from top chunk
        transcript_text = chunks[0].payload['combined_text'] if chunks else "*No transcript found.*"

        try:
            video_paths = extract_video_clips(chunks)
            if not video_paths:
                output += "\n\n*Note: No video clips could be extracted. This may be due to missing video files.*"
        except Exception as e:
            print(f"Error in video extraction: {e}")
            video_paths = []
            output += f"\n\n*Note: Error extracting video clips: {str(e)}*"

        # Replace the interim message with the final response
        message_history[-1] = {"role": "assistant", "content": output}

        return message_history, video_paths[0] if video_paths else None, transcript_text

    except Exception as e:
        print(f"[ERROR] Exception in gradio_chatbot(): {e}")
        message_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
        return message_history, None, "*Transcript unavailable due to an error.*"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 RAG Video Assistant")
    chatbot = gr.Chatbot(label="Ask a question about the lecture", type="messages")
    textbox = gr.Textbox(placeholder="Ask about CNNs, ResNets, or cross-entropy...")

    video_display = gr.Video(label="Relevant Clip")
    transcript_display = gr.Markdown(label="Transcript")

    def user_submit(message, history):
        history.append({"role": "user", "content": message})
        return "", history

    textbox.submit(user_submit, [textbox, chatbot], [textbox, chatbot]).then(
        gradio_chatbot, chatbot, [chatbot, video_display, transcript_display]
    )

demo.launch()