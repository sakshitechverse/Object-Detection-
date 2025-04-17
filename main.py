import stremlit as st 
from transformers import YolosImages Processor , YoloObjectDetection
from PIL import image, ImageDraw
import torch

#load the moodel and image processor 
model = YoosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

st.title("YOLOs object Detection with Bounding Boxes")

#Upload image
upload_file  st.file_uploader("Upload an image", type =["jpg","jpeg","png"])

if uploaded_file is not None:

  #display uploaded image
  image -Image.open(uploaded_file)
  st.image(image,cption = 'Uploaded Image', use_column_width=True)

   # Process the image and run object detection
    st.write("Detecting objects...")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

#process results
target_sizes = torch.tensor([images.size [::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

#derew bounding boxs on the image
draw = ImageDraw.Draw(image)
for score, labe, box in zip(results["scores"],results["labels"],results["boxes"]):
  box = [round(i,2) for i in box.tolist()]
  draw.rectangle(box, outline="red", width=3)
  draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")

#Display the image with boundind boxes 
st.image, caption = 'detected objets ' , use _column_width = true )

 # Optionally, display detection results in text form
    st.write("Detection Results:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        st.write(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")


