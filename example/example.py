import cv2
from gym_exercise_processor.real_time_processor import DefaultProcessor

if __name__ == "__main__":
    model_path = "checkpoint.pth"

    processor = DefaultProcessor(model_path)

    cap = cv2.VideoCapture(0)  # Use webcam, or replace with video file path
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        data = processor.process_frame_to_json(frame)

        # visualizing json
        if data["exercise_type"]:
            cv2.putText(frame, f"Exercise: {data['exercise_type']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {data['reps']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Correct Reps: {data['correct_reps']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if data["stage"]:
                cv2.putText(frame, f"Stage: {data['stage']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Exercise: None", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        landmarks = data["landmarks"]
        for landmark in landmarks:
            if landmark["visibility"] > 0.5:
                x = int(landmark["x"] * frame.shape[1])
                y = int(landmark["y"] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        connections = data["connections"]
        for conn in connections:
            start = conn["start"]
            end = conn["end"]
            if start < len(landmarks) and end < len(landmarks):
                x1 = int(landmarks[start]["x"] * frame.shape[1])
                y1 = int(landmarks[start]["y"] * frame.shape[0])
                x2 = int(landmarks[end]["x"] * frame.shape[1])
                y2 = int(landmarks[end]["y"] * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
        cv2.imshow('Exercise Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    processor.release()