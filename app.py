import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --------------------
# 1. App Configuration
# --------------------
st.set_page_config(page_title="Species Predictor", layout="centered")
st.title("Image-Based Fish Species Predictor")
st.markdown("Upload an image of a fish, and I'll predict the species from a specific list of options.")
st.write("---")

# --------------------
# 2. Model and Class Names Loading
# --------------------
# Use st.cache_data to load the model and processor only once.
@st.cache_data(show_spinner=False)
def load_model():
    """Loads a pre-trained CLIP model and its processor for zero-shot classification."""
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
    return processor, model

# Your specific list of class names
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Load the model with a spinner and error handling.
try:
    with st.spinner('Loading the deep learning model... this might take a moment.'):
        processor, model = load_model()
    st.info("Model loaded successfully. Ready for predictions!")
except Exception as e:
    st.error(f"❌ An error occurred while loading the model: {e}")
    st.error("This is likely due to the model being too large for the hosting environment. "
             "The app cannot run without the model. Please consider using a service with more memory.")
    st.stop()


# --------------------
# 3. User Interface for Image Upload
# --------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a fish to get a prediction from a predefined list."
)

# --------------------
# 4. Prediction Logic
# --------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Predicting...")

    try:
        # Open and prepare the image.
        image = Image.open(uploaded_file).convert("RGB")

        # Prepare the image and text inputs for the model.
        inputs = processor(
            text=class_names,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Make the prediction.
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the scores for each class.
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Find the class with the highest probability.
        predicted_class_idx = probs.argmax().item()
        confidence = probs[0][predicted_class_idx].item()
        
        # Display the result.
        predicted_species = class_names[predicted_class_idx]
        st.write(f"I predict this is a: **{predicted_species}** with a confidence of {confidence:.2f}")
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
