
import streamlit as st
import os
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
#from dotenv import load_dotenv

# Load environment variables from the .env file
#load_dotenv()

# Access the token
#HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]


# Title and description
st.title("Image Caption Creation")
st.write("This application demonstrates the BLIP-2 model to generate an suitable caption for the images uploaded.")

# Manage temporary cache folder
hf_cache_dir = "./hf_cache"
os.environ["HF_HOME"] = hf_cache_dir

# Ensure enough disk space
if not os.path.exists(hf_cache_dir):
    os.makedirs(hf_cache_dir)

# Load model and processor
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",  # Smaller valid
    token=HUGGING_FACE_TOKEN,
    cache_dir=hf_cache_dir
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="./offload",  # Specify an offload folder
    token=HUGGING_FACE_TOKEN,
    cache_dir=hf_cache_dir
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# File upload for input image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image

    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Generate a caption
    st.write("Generating caption...")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Display the generated caption
    st.subheader("Generated Caption")
    st.write(caption)
else:
    st.info("Please upload an image to generate a caption.")