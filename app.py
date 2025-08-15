import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --------------------
# 1. App Configuration
# --------------------
st.set_page_config(page_title="Fish Predictor", layout="centered")
st.title("Image-Based Species Predictor")
st.markdown("This is a new, standalone version of the app using a lightweight model to avoid system errors. "
            "It predicts common objects from a broad list, not your specific fish species.")
st.write("---")

# --------------------
# 2. Model and Processor Loading
# --------------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor():
    """Loads a pre-trained image classification model and its processor."""
    model_name = "google/vit-tiny-patch16-224"
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("The app cannot run without the model. Please check your internet connection or try again later.")
        return None, None

try:
    with st.spinner('Loading the deep learning model... This is a very small model, so it should be fast.'):
        processor, model = load_model_and_processor()
    if processor and model:
        st.success("✅ Model loaded successfully. Ready for predictions!")
    else:
        st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# --------------------
# 3. User Interface for Image Upload
# --------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to get the top 5 predictions from the model."
)

# --------------------
# 4. Prediction Logic
# --------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")

    try:
        # Open and prepare the image.
        image = Image.open(uploaded_file).convert("RGB")
        
        # Preprocess the image for the model.
        inputs = processor(images=image, return_tensors="pt")
        
        # Make the prediction.
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
        st.error(f"An error occurred during prediction: {e}")
