import cv2
import numpy as np
import os

class YOLODetector:
    def __init__(self, config_path, weights_path, classes_path):
        """
        Initialize YOLO detector with model files
        """
        # Load YOLO model
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Use GPU if available (much faster)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Confidence and NMS thresholds
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def detect_objects(self, image_path):
        """
        Detect objects in the given image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        height, width, channels = img.shape
        
        # Create blob from image (normalize, resize, scale)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Pass blob through network
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes, confidences, class_ids = self._process_outputs(outputs, width, height)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Draw bounding boxes
        result_img = self._draw_boxes(img, boxes, confidences, class_ids, indices)
        
        return result_img, len(indices)
    
    def _process_outputs(self, outputs, width, height):
        """
        Process YOLO outputs to extract bounding boxes, confidences, and class IDs
        """
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter weak detections
                if confidence > self.confidence_threshold:
                    # Scale bounding box coordinates to original image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def _draw_boxes(self, img, boxes, confidences, class_ids, indices):
        """
        Draw bounding boxes and labels on the image
        """
        result_img = img.copy()
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Draw rectangle
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                
                # Create label text
                label_text = f"{label}: {confidence:.2f}"
                
                # Calculate label background size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(result_img, (x, y - text_height - baseline - 5), 
                             (x + text_width, y), color, -1)
                
                # Draw label text
                cv2.putText(result_img, label_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_img

def download_yolo_files():
    """
    Download required YOLO files if they don't exist
    """
    import urllib.request
    
    files = {
        'yolov4.cfg': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg',
        'yolov4.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")

def main():
    """
    Main function to run object detection
    """
    print("YOLO Object Detection System")
    print("=" * 40)
    
    # Download required files
    download_yolo_files()
    
    # Initialize YOLO detector
    try:
        detector = YOLODetector('yolov4.cfg', 'yolov4.weights', 'coco.names')
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    while True:
        print("\nOptions:")
        print("1. Detect objects in an image")
        print("2. Exit")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '2':
            print("Goodbye!")
            break
        elif choice == '1':
            image_path = input("Enter the path to your image: ").strip()
            
            if not os.path.exists(image_path):
                print("Error: Image file not found!")
                continue
            
            # Detect objects
            result_img, num_detections = detector.detect_objects(image_path)
            
            if result_img is not None:
                print(f"Detected {num_detections} objects!")
                
                # Display result
                cv2.imshow('Object Detection Results', result_img)
                print("Press any key in the image window to close...")
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                
                # Ask if user wants to save the result
                save_choice = input("Do you want to save the result? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_path = "detection_result.jpg"
                    cv2.imwrite(output_path, result_img)
                    print(f"Result saved as {output_path}")
        else:
            print("Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()