"""Optional Streamlit interface for the shared classifier."""

import streamlit as st

from server import util


def display_name(label):
    return label.replace("_", " ").title()


st.set_page_config(page_title="Sports Person Classifier")
st.title("Sports Person Classifier")
st.write("Explore model predictions for up to eight qualifying faces in a photo.")
uploaded = st.file_uploader(
    "Choose a photo",
    type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=util.MAX_IMAGE_BYTES // (1024 * 1024),
)

if uploaded is not None:
    try:
        image = util.decode_image(uploaded.getvalue())
        result = util.classify_faces(image)
    except util.ImageInputError as error:
        st.error(str(error))
    except util.ClassifierUnavailable:
        st.error("The classifier is temporarily unavailable.")
    else:
        st.image(image, channels="BGR", caption="Uploaded photo")
        faces = result["faces"]
        if not faces:
            st.info(
                "No face with two visible eyes was detected. Try a clear, front-facing photo."
            )
        else:
            if result["detected_faces"] > len(faces):
                st.info(
                    f"Showing {len(faces)} of {result['detected_faces']} detected faces."
                )
            tabs = st.tabs([f"Face {index}" for index in range(1, len(faces) + 1)])
            for tab, face in zip(tabs, faces, strict=True):
                with tab:
                    box = face["box"]
                    x, y, width, height = (
                        box[key] for key in ("x", "y", "width", "height")
                    )
                    st.image(
                        image[y : y + height, x : x + width], channels="BGR", width=150
                    )
                    st.subheader(display_name(face["label"]))
                    for entry in face["scores"]:
                        label, score = display_name(entry["label"]), entry["score"]
                        st.progress(score / 100, text=f"{label} · {score:.2f}%")

st.caption(
    "The model ranks five athletes. Its scores cannot verify someone’s identity."
)
