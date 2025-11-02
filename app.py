import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import zipfile

# ----------------------------
# 1. Page setup
# ----------------------------
st.set_page_config(page_title="AI Recipe Generator 🍲", layout="centered")
st.title("🍳 AI Recipe Generator")
st.write("Generate creative recipes using your fine-tuned GPT-2 model!")

# ----------------------------
# 2. Unzip and load model
# ----------------------------
MODEL_DIR = "Model"

if not os.path.exists(MODEL_DIR):
    zip_path = "Model.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        st.success("✅ Model extracted successfully.")
    else:
        st.error("❌ Model zip not found. Please upload `gpt2_recipes.zip` to the repo.")
        st.stop()

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# 3. Text generation function
# ----------------------------
def generate_recipe(prompt, max_length=300, temperature=0.8, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=False)

# ----------------------------
# 4. Streamlit interface
# ----------------------------
title = st.text_input("Enter a recipe title:", "Chocolate Cake")
max_len = st.slider("Max length of recipe", 100, 600, 300, 50)
temp = st.slider("Creativity (temperature)", 0.5, 1.5, 0.9, 0.1)
top_p = st.slider("Top-p sampling", 0.5, 1.0, 0.9, 0.05)

if st.button("🍰 Generate Recipe"):
    with st.spinner("Cooking up your recipe..."):
        prompt = f"<|title|>{title}<|ingredients|>"
        output = generate_recipe(prompt, max_length=max_len, temperature=temp, top_p=top_p)

        # Split the recipe for nicer formatting
        if "<|ingredients|>" in output and "<|steps|>" in output:
            parts = output.split("<|ingredients|>")[1].split("<|steps|>")
            ingredients = parts[0].strip()
            steps = parts[1].strip()
        else:
            ingredients, steps = "N/A", output.strip()

        st.subheader("🧂 Ingredients")
        st.write(ingredients)

        st.subheader("👩‍🍳 Steps")
        st.write(steps)

st.markdown("---")
st.caption("Built with 🤗 Transformers & Streamlit")
