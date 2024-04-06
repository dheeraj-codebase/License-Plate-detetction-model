import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# page configurations
st.title("License Plate Detection Application")
license_plate_model = YOLO('models/license_plate_detector.pt')
car_model = YOLO('yolov8n.pt')
vehicles = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


# Upload image function
def upload_image():
    uploaded_file = st.file_uploader("Upload Image to extract the license plates", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Save the uploaded image to a local file
            with open("uploaded_image.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success("Image uploaded successfully and is being processed....")
            image_path = "uploaded_image.jpg"
            image = cv2.imread(image_path)
            plate_results = license_plate_model(image)
            annotated_frame = plate_results[0].plot()
            car_results = car_model(image)[0]
            detections_ = []
            for detection in car_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score, int(class_id)])

            annotated_frame = annotated_frame.copy()  # Use the original image
            for bbox in detections_:
                x1, y1, x2, y2, score, class_id = bbox
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Add text annotation (class label and score) for car
                label = f"{vehicles[class_id]}: {score:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            st.image(annotated_frame, caption="Selected Image", use_column_width=True)


# Upload video function
def upload_video():
    uploaded_file = st.file_uploader("Upload Video to extract the license plates", type=["mp4"])
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Save the uploaded video to a local file
            with open("uploaded_video.mp4", "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success("Video is being processed....")
            video_path = "uploaded_video.mp4"
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Define the codec and create VideoWriter object
            out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 inference for license plate detection on the frame
                    plate_results = license_plate_model(frame)

                    # Visualize the results on the frame for license plate detection
                    annotated_frame = plate_results[0].plot()

                    # Run YOLOv5 inference for car detection on the frame
                    car_results = car_model(frame)[0]
                    detections_ = []

                    # Iterate through the detections and filter based on vehicles
                    for detection in car_results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles:
                            detections_.append([x1, y1, x2, y2, score, int(class_id)])

                    # Visualize the results on the frame for car detection
                    new_frame = annotated_frame.copy()  # Use the original frame
                    for bbox in detections_:
                        x1, y1, x2, y2, score, class_id = bbox
                        # Draw bounding box
                        cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # Add text annotation (class label and score) for car
                        label = f"{vehicles[class_id]}: {score:.2f}"
                        cv2.putText(new_frame, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    out.write(new_frame)

                    # Display the annotated frames
                    cv2.imshow("License Plate Detection", new_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object, release the output video object, and close the display window
            cap.release()
            out.release()
            cv2.destroyAllWindows()


upload_image()
upload_video()

