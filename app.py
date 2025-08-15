import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# --------------------
# 1. App Configuration
# --------------------
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("General Image Classifier")
##st.markdown("This app uses the **`google/vit-base-patch16-224`** model to classify uploaded images.")
st.write("---")

# --------------------
# 2. Model and Processor Loading
# --------------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor():
    """Loads a pre-trained Vision Transformer (ViT) model and its processor."""
    model_name = "google/vit-base-patch16-224"
    try:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("Please ensure the model name is correct and you have an internet connection.")
        return None, None

try:
    with st.spinner('Loading the deep learning model...'):
        processor, model = load_model_and_processor()
    if processor and model:
        st.success("✅ Model loaded successfully. Ready for classification!")
    else:
        st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during model loading: {e}")
    st.stop()

# --------------------
# 3. User Interface for Image Upload
# --------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to get predictions."
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

        # Get the top predictions.
        logits = outputs.logits
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
