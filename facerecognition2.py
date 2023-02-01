import cv2
import requests


class FaceIdentifier:
    def __init__(self, camera_url, model_path):
        self.camera = cv2.VideoCapture(camera_url)
        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trained_model.yml")

    def identify(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.camera.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                id, conf = self.recognizer.predict(roi_gray)

                if conf < 50:
                    name = requests.get(f"http://example.com/api/employees/{id}").json()["name"]
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the resulting frame
                    cv2.imshow("Face Identifier", frame)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the camera and destroy the windows
            self.camera.release()
            cv2.destroyAllWindows()
