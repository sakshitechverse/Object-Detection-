import streamlit as st
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

# Load the model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

st.title("YOLOS Object Detection with Bounding Boxes")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image and run object detection
    st.write("Detecting objects...")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Process results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")

    # Display the image with bounding boxes
    st.image(image, caption='Detected Objects', use_column_width=True)

    # Optionally, display detection results in text form
    st.write("Detection Results:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        st.write(f"Detected **{model.config.id2label[label.item()]}** "
                 f"with confidence **{round(score.item(), 3)}** "
                 f"at location **{[round(i, 2) for i in box.tolist()]}**")
