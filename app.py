import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --------------------
# 1. App Configuration
# --------------------
st.set_page_config(page_title="Fish Species Classifier", layout="centered")
st.title("Image-Based Fish Species Classifier")
st.markdown("This app uses the **`srihari19/fish-classification`** model, which is specifically trained to identify different types of fish.")
st.write("---")

# --------------------
# 2. Model and Processor Loading
# --------------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor():
    """Loads the fish classification model and its processor."""
    model_name = "srihari19/fish-classification"
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("Please ensure the model name is correct and you have an internet connection.")
        return None, None

try:
    with st.spinner('Loading the deep learning model... This might take a moment.'):
        processor, model = load_model_and_processor()
    if processor and model:
        st.success("✅ Model loaded successfully. Ready for classification!")
    else:
        st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# --------------------
# 3. User Interface for Image Upload
# --------------------
uploaded_file = st.file_uploader(
    "Choose an image of a fish...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to get the top 5 predictions for the fish species."
)

# --------------------
# 4. Classification Logic
# --------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    try:
        # Open and prepare the image.
        image = Image.open(uploaded_file).convert("RGB")
        
        # Preprocess the image for the model.
        inputs = processor(images=image, return_tensors="pt")
        
        # Make the classification.
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the scores (logits) for each class.
        logits = outputs.logits
        
        # Convert logits to probabilities and get the top 5 predictions.
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        top_5_probs, top_5_indices = torch.topk(probabilities, 5)

        # Display the results in a formatted list.
        st.subheader("Top 5 Predictions:")
        for i in range(5):
            predicted_label = model.config.id2label[top_5_indices[i].item()]
            confidence = top_5_probs[i].item()
            st.write(f"**{i + 1}.** **{predicted_label}** with a confidence of {confidence:.2f}")

        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
