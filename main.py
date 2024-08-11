import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Ensure models are downloaded before running the app
import download_models

# Function to load the selected model
def load_model(model_name):
    model_path = os.path.join("models", model_name)
    model = YOLO(model_path)
    return model

# Function to perform inference based on model selection
def perform_inference(model, uploaded_image):
    image = Image.open(uploaded_image)
    results = model(image)
    return results

# Streamlit app
def main():
    st.title("Barcode, QR Code, and Vehicle Detection Application")

    # Step 1: Task selection
    task = st.sidebar.radio("Select Detection Task", ("Barcode-QR Detection", "Vehicle Detection"))

    # Step 2: Model selection based on the selected task
    model_choice = st.sidebar.radio("Select Model Version", ("YOLOv8", "YOLOv9", "YOLOv10"))

    # Correctly form the model name
    model_name = f"{model_choice.lower()}_{'barcode' if task == 'Barcode-QR Detection' else 'vehicle'}.pt"

    # Load the selected model
    model = load_model(model_name)

    # Step 3: Image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Run inference based on selected model
        if st.button("Run Detection"):
            with st.spinner('Running inference...'):
                try:
                    # Perform inference
                    results = perform_inference(model, uploaded_image)

                    # Visualize and save results
                    for i, r in enumerate(results):
                        st.subheader("Result")
                        
                        # Plot results image
                        im_bgr = r.plot()  # BGR-order numpy array
                        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                        st.image(im_rgb, caption="Result", use_column_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
