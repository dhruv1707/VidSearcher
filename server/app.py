import whisper
from moviepy.editor import VideoFileClip
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from db.search_engine import add, query
from PIL import Image
from image.image_caption import process_image,refine_caption
import tempfile
import torch

whisper_model = whisper.load_model("large-v2")

def extractFrames(video_path, interval=2):
    frames = []
    video = VideoFileClip(video_path)
    duration = video.duration
    frame_time = 0
    while frame_time < duration:
        frame = video.get_frame(frame_time)
        image = Image.fromarray(frame)
        frames.append((image, frame_time))
        frame_time += interval
    return frames

def process_video(video_path, fragment_duration=2):
    video = VideoFileClip(video_path)
    frames = extractFrames(video_path=video_path, interval=fragment_duration)
    fragment_start = 0
    duration = video.duration
    for frame, time in frames:
        description = process_image(frame=frame)
        refined_caption = refine_caption(description)
        print(f"Refined Caption: {refined_caption}")
        fragment_start = time
        end_time = min(fragment_start + fragment_duration, duration)
        video_fragment = video.subclip(fragment_start, end_time)
        if video_fragment.audio:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
                video_fragment.audio.write_audiofile(temp_audio_file.name)
                transcription = whisper_model.transcribe(temp_audio_file.name)
                print(transcription, fragment_start)
                concatenated_description = f"{transcription['text']}{refined_caption}"
                add(concatenated_description, video_path=video_path, start_time=time)
        else:
            transcription = ""
            concatenated_description = f"{refined_caption}"
            add(concatenated_description, video_path=video_path, start_time=time)

def process_input(directory_path):
    print(directory_path)
    for root, dirs, files in os.walk(directory_path):
        print(f"Directories: {dirs}")
        for file in files:
            if file.endswith((".mov", ".mp4", ".avi")):
                file_path = os.path.join(root, file)
                print(file_path)
                process_video(file_path)
                print(file)
desktop_path = "~/Desktop/Beerbiceps - Assignments/LA4/Resources"
full_path = os.path.expanduser(desktop_path)
print("Processing the input")
process_input(full_path)
print("Processed the input")
        



def extract_frames_from_directory(input_dir, output_base_dir, num_frames):
    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.mov', '.mp4', '.avi')):  # Check for valid video file extensions
            video_path = os.path.join(input_dir, filename)
            # Create an output directory for the frames of this specific video
            output_dir = os.path.join(output_base_dir, os.path.splitext(filename)[0])  # Remove file extension for folder name
            # Extract frames from the video
            extractFrames(video_path, output_dir, num_frames)
            print(f"Extracted frames from {filename} to {output_dir}")

output_dir = "/Users/dhruvmehrottra007/Desktop/DejaVu/content/frames"
extract_frames_from_directory(output_video_file, output_dir, 3)

model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model.eval()

text = "[CLS] An image showing [MASK]."
encoding = tokenizer(text, 
                     padding="max_length",
                     max_length = 10,
                     truncation = True,
                     return_tensors = "pt",
                     )
print("Vocabulary Size:", tokenizer.vocab_size)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

modelVision = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"

projection_layer = nn.Linear(in_features=512, out_features=2048)

for filename in os.listdir(output_dir):
    fragment_path = os.path.join(output_dir, filename)
    if os.path.isdir(fragment_path):
        for frame_file in os.listdir(fragment_path):
            if frame_file.endswith((".png", ".jpg", ".jpeg")):
                print(frame_file, fragment_path)
                frame_path = os.path.join(fragment_path, frame_file)
                image = Image.open(frame_path).convert("RGB")
                print(image)
                images_processed = processor(image, return_tensors = "pt")
               # image_tensor = transform(image).unsqueeze(0)
                print(images_processed)
                with torch.no_grad():
                    features = modelVision.get_image_features(**images_processed)
                    print(features.shape)
                    visual_attention_mask = torch.ones(features.shape[:-1], dtype=torch.float)
                    print(visual_attention_mask.shape, attention_mask.shape)
                    visual_attention_mask = visual_attention_mask.unsqueeze(1).expand(-1, attention_mask.size(1))
                    visual_projected_features = projection_layer(features)
                    visual_projected_features = visual_projected_features.unsqueeze(1)
                    visual_attention_mask = torch.ones(visual_projected_features.shape[:-1], dtype=torch.float)
                    
                    outputs = model(input_ids = input_ids, attention_mask=attention_mask, visual_embeds = visual_projected_features, visual_attention_mask=visual_attention_mask)

                print(outputs.prediction_logits)

                mask_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]  # Get index of [MASK]
                masked_logits = outputs.prediction_logits[0, mask_position]
                print(masked_logits)
                masked_probs = torch.softmax(masked_logits, dim=-1)  # Apply softmax to get probabilities
                predicted_token_id = masked_probs.argmax(dim=-1)  # Get the predicted token ID
                predicted_token = tokenizer.decode(predicted_token_id)
                print(predicted_token)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

search_query = "a woman in a bikni"
fragment_to_similarity = {}
for filename in os.listdir(output_dir):
    fragment_path = os.path.join(output_dir, filename)
    if os.path.isdir(fragment_path):
        for frame_file in os.listdir(fragment_path):
            results = []
            if frame_file.endswith((".png", ".jpg", ".jpeg")):
                print(frame_file, fragment_path)
                frame_path = os.path.join(fragment_path, frame_file)
                image = Image.open(frame_path).convert("RGB")
                text_inputs = clip_processor(text=[search_query], return_tensors="pt", padding=True, truncation=True)
                print(image)
                image_inputs = clip_processor(images = image, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    text_embeds = modelVision.get_text_features(**text_inputs)
                    image_embeds = modelVision.get_image_features(**image_inputs)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                similarity = torch.matmul(text_embeds, image_embeds.T)
                print(f"Similarity score: {similarity.item()}")
                results.append(similarity.item())

        fragment_to_similarity[filename] = sum(results)/len(results)

print(fragment_to_similarity)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

fragments_to_captions = defaultdict(list)

for filename in os.listdir(output_dir):
    fragment_path = os.path.join(output_dir, filename)
    if os.path.isdir(fragment_path):
        for frame_file in os.listdir(fragment_path):
            if frame_file.endswith((".png", ".jpg", ".jpeg")):
                frame_path = os.path.join(fragment_path, frame_file)
                image = Image.open(frame_path).convert("RGB")
                inputs = processor(image, return_tensors="pt")
                out = model.generate(**inputs, num_beams = 6, max_length=30, early_stopping=True)
                caption = processor.decode(out[0], skip_special_tokens=True)
                print(caption, frame_file, filename)
                fragments_to_captions[filename].append(caption)

print(fragments_to_captions)

unique_captions = {}
for fragment in fragments_to_captions:
    captions_embeddings = modelSentence.encode(fragments_to_captions[fragment], convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(captions_embeddings, captions_embeddings)
    print(similarity_matrix)
    unique_captions[fragment] = []
    for i in range(len(fragments_to_captions[fragment])):
        if all(similarity_matrix[i][j] < 0.7 for j in range(len(unique_captions))):
            unique_captions[fragment].append(fragments_to_captions[fragment][i])
unique_captions = fragments_to_captions

print(len(unique_captions["fragment_3"]), len(fragments_to_captions["fragment_3"]))


model_gemini = genai.GenerativeModel('gemini-1.5-flash')

response_by_fragment = {}

for fragment in unique_captions:
    captions_fragment = unique_captions[fragment]
    combined_captions = " ".join(captions_fragment)
    response_by_fragment[fragment] = []
    print(combined_captions)
    prompt = f"You're a bot that's great at summarizing a bunch of captions generated for a fragment of the video. Keep in the mind the captions combined together may not make sense. It is your job to create a summary out of all of these combined captions that makes sense and accurately represents that fragment of the video. Each of the independent captions represent a frame of that scene/fragment. Here are the combined captions: {combined_captions}"
    response = model_gemini.generate_content(prompt, generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.6,))
    response_by_fragment[fragment].append(response.text)
    print(response.text)

print(response_by_fragment)


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

for filename in os.listdir(output_dir):
    fragment_path = os.path.join(output_dir, filename)
    if os.path.isdir(fragment_path):
        for frame_file in os.listdir(fragment_path):
            if frame_file.endswith((".png", ".jpg", ".jpeg")):
                print(frame_file, fragment_path)
                frame_path = os.path.join(fragment_path, frame_file)
                image = Image.open(frame_path).convert("RGB")
                print(image)
                pixel_values = image_processor(image, return_tensors="pt").pixel_values

                generated_ids = model.generate(pixel_values,max_length=20, num_beams=6)
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(generated_text)


input_video_file = "/Users/dhruvmehrottra007/Desktop/selza/Res/Brolls/IMG_4308.mp4"
output_video_file = "/Users/dhruvmehrottra007/Desktop/DejaVu/content/video_fragments/"
os.makedirs(output_video_file, exist_ok=True)
audio_output_directory = "/Users/dhruvmehrottra007/Desktop/DejaVu/content/audio_fragments/"


def splitVideoFragments(input_video_file, fragment_duration=2):
    print(f"Processing input video file: {input_video_file}") 
    video = VideoFileClip(input_video_file)
    fragment_start = 0
    duration = video.duration 
    print(duration)
    fragment_end = fragment_duration
    fragment_num = 0
    while fragment_start <= duration:
        output_file = os.path.join(output_video_file, f"fragment_{fragment_num}.mov")
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Overwriting.")  
            os.remove(output_file)  
        ffmpeg_extract_subclip(input_video_file, fragment_start, min(fragment_end, duration), output_file)
        fragment_start = fragment_end
        fragment_end += fragment_duration
        fragment_num += 1

splitVideoFragments(input_video_file=input_video_file, fragment_duration=1)

files = [name for name in os.listdir(output_video_file) if os.path.isfile(os.path.join(output_video_file, name)) and os.path.join(output_video_file, name).endswith(('.mov', '.mp4', '.avi'))]
total_files = len(files)
print(total_files, files)
transcriptions = []
fileNames = []
for i in range(1, total_files+1):
    video_path = os.path.join(output_video_file, f"fragment_{i}.mov")
    audio_output_path = os.path.join(audio_output_directory, f"fragment_{i}.mp3")
    video = VideoFileClip(video_path)

    audio = video.audio
    audio.write_audiofile(audio_output_path)
    result = whisper_model.transcribe(audio_output_path)
    result = result["text"].strip()
    if result == "":
        transcriptions.append("")
    else:
        transcriptions.append(result)
    fileNames.append(f"fragment_{i}.mov")

df = pd.DataFrame({"File Name": fileNames, "Transcription": transcriptions})

print(df)
df.to_csv("df.csv")
transcriptions_map = df.set_index("Transcription")["File Name"].to_dict()

modelSentence = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = modelSentence.encode(transcriptions)
embeddings = {transcriptions_map[transcription]: embedding for transcription, embedding in zip(transcriptions, embeddings)}

#reducer = umap.UMAP()
#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(list(embeddings.values()))
#reduced_data = reducer.fit_transform(scaled_data)

user_input = input("Enter the search term:")
user_input_embedding = modelSentence.encode(user_input)
user_input_embedding_reshaped = user_input_embedding.reshape(1,-1)
similarities = cosine_similarity(user_input_embedding_reshaped, np.array(list(embeddings.values())))
