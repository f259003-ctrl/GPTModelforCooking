import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import gdown
import zipfile

# ===============================
# CONFIGURATION
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=1GzwppdOLfOto3dcUJAN-tADwdWL69yYF"  # converted to direct-download link
MODEL_DIR = "gpt2-recipes"
ZIP_PATH = "gpt2-recipes.zip"

# ===============================
# DOWNLOAD & EXTRACT MODEL
# ===============================
@st.cache_resource
def download_and_load_model():
    # Download from Google Drive if not already present
    if not os.path.exists(MODEL_DIR):
        st.info("🔽 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, ZIP_PATH, quiet=False)

        # Extract the zip file
        st.info("📦 Extracting model files...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    # Load tokenizer and model
    st.info("⚙️ Loading model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


model, tokenizer = download_and_load_model()

# ===============================
# STREAMLIT INTERFACE
# ===============================
st.title("🍳 AI Recipe Generator")
st.write("Generate creative recipes using your fine-tuned GPT-2 model!")

prompt = st.text_area("🧂 Enter ingredients or a recipe idea:", "chicken, garlic, rice")
max_length = st.slider("📏 Max recipe length (tokens):", 50, 300, 150)
temperature = st.slider("🔥 Creativity (temperature):", 0.5, 1.5, 1.0)
top_p = st.slider("🎯 Top-p (nucleus sampling):", 0.5, 1.0, 0.9)

if st.button("🍽️ Generate Recipe"):
    if prompt.strip() == "":
        st.warning("Please enter a recipe idea or ingredients.")
    else:
        with st.spinner("Cooking up your recipe... 🍲"):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            # Generate text
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            recipe = generated_text[len(prompt):].strip()

            st.subheader("✨ Generated Recipe:")
            st.write(recipe)

st.markdown("---")
st.caption("Model fine-tuned on custom recipe dataset using GPT-2.")
