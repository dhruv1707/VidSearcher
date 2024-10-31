import numpy as np
from PIL import Image
from transformers import VisualBertForPreTraining, AutoTokenizer
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration
import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from torchvision import transforms
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

load_dotenv()
api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)

t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

previous_captions = []

def process_image(frame):
    print(type(frame))
    image = frame.convert("RGB")
    inputs = processor2(image, return_tensors="pt").to(dtype = torch.bfloat16)
    out = model2.generate(**inputs)
    caption = processor2.batch_decode(out, skip_special_tokens=True)[0].strip()
    print(caption)
    return caption


def refine_caption(caption):
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"You're a bot that's great at refining a caption generated for a frame of the video. Keep in the mind the caption may not make sense or may be unintelligible. Also many times you may not even have the full information to make a good judgement. It is your job to create a refined caption that makes sense and accurately represents that frame of the video. Each of the independent captions represent a frame that's extracted every 2 seconds in the video. Here's the caption: {caption}. Kindly don't include any othe information or any other introductory text."
    response = model_gemini.generate_content(prompt, generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.6,))
    return response.text

