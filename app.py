import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model with caching for better performance
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return pipe

# Initialize the model
pipe = load_model()

# Streamlit UI
st.title("AI-Powered Image Generator")
st.write("Enter a prompt to generate an AI-generated image.")

# User input
prompt = st.text_input("Describe your image:")

if st.button("Generate"):
    with st.spinner("Creating image..."):
        image = pipe(prompt).images[0]  # Generate the image
        image.save("generated_image.png")
        st.image(image, caption="Generated Image", use_column_width=True)
