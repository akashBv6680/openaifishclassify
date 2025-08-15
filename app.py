import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --------------------
# 1. App Configuration
# --------------------
st.set_page_config(page_title="Zero-Shot Image Classifier", layout="centered")
st.title("Zero-Shot Image Classifier")
st.markdown("This app uses a **zero-shot** model (`openai/clip-vit-base-patch32`), which means it can classify an image based on **any text labels you provide**.")
st.write("---")

# --------------------
# 2. Model and Processor Loading
# --------------------
@st.cache_resource(show_spinner=False)
def load_clip_model():
    """Loads a pre-trained CLIP model and its processor."""
    model_name = "openai/clip-vit-base-patch32"
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("Please ensure the model name is correct and you have an internet connection.")
        return None, None

try:
    with st.spinner('Loading the deep learning model...'):
        processor, model = load_clip_model()
    if processor and model:
        st.success("✅ Model loaded successfully. Ready for classification!")
    else:
        st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# --------------------
# 3. User Interface for Image and Text Input
# --------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to get predictions."
)

st.subheader("Provide Classification Labels")
labels_text = st.text_input(
    "Enter a comma-separated list of labels (e.g., a photo of a dog, a photo of a cat, a photo of a fish)",
    value="a photo of a fish, a photo of a bird, a photo of a boat, a photo of a car"
)

# --------------------
# 4. Classification Logic
# --------------------
if uploaded_file is not None and labels_text:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    try:
        # Open and prepare the image and labels.
        image = Image.open(uploaded_file).convert("RGB")
        candidate_labels = [label.strip() for label in labels_text.split(',')]
        
        # Preprocess the image and labels for the model.
        inputs = processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Make the classification.
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the scores (logits) and calculate probabilities.
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Get the top 5 predictions.
        top_5_probs, top_5_indices = torch.topk(probs, 5, dim=1)

        # Display the results in a formatted list.
        st.subheader("Top 5 Predictions:")
        for i in range(top_5_probs.size(1)):
            label_index = top_5_indices[0][i].item()
            predicted_label = candidate_labels[label_index]
            confidence = top_5_probs[0][i].item()
            st.write(f"**{i + 1}.** **{predicted_label}** with a confidence of {confidence:.2f}")

        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
