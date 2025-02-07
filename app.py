import streamlit as st
import base64
from PIL import Image
import io
from server import util
from server import wavelet

# Load the model
util.load_saved_artifacts()

# Title and description of the app
st.title("Sports Person Classifier")
st.markdown("Upload an image of a sports person to classify it.")

# File uploader for image
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image uploaded by the user
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert the uploaded image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Add base64 prefix to the string to ensure it's valid
    img_str = f"data:image/png;base64,{img_str}"

    # Call the classification function from util
    response = util.classify_image(img_str)  # Send base64 string instead of file path

    if not response:
        st.error("Can't classify image. Classifier was not able to detect face and two eyes properly.")
    else:
        # Show classification result
        st.subheader("Classification Result:")
        st.write(f"Predicted Class: {response[0]['class']}")

        # Show probability scores for each player
        st.subheader("Class Probability Scores:")
        class_names = ['lionel_messi', 'maria_sharapova', 'roger_federer', 'serena_williams', 'virat_kohli']
        for class_name in class_names:
            score = response[0]['class_probability'][response[0]['class_dictionary'][class_name]]
            st.write(f"{class_name.replace('_', ' ').title()}: {score:.2f}%")

# Styling the page
st.markdown(
    """
    <style>
    .css-1aumxhk {
        background-color: #f8f9fa;
    }
    .css-1k0dwgp {
        color: #4CAF50;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)
