import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press Q to quit.")

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Flip the image horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Display instructions
    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show the frame
    cv2.imshow("AI Sign Language Translator", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()








'''''
1. Import OpenCV
import cv2

This imports the OpenCV library so we can use its functions.

2. Open the webcam
cap = cv2.VideoCapture(0)

VideoCapture() tells OpenCV to connect to a camera.

The 0 means:

0 → Default webcam
1 → Second camera (if connected)
2 → Third camera

cap is an object that manages communication with the camera.

3. Check if the camera opened
if not cap.isOpened():

This checks whether the webcam was opened successfully.

If not:

print("Error: Could not open webcam.")
exit()

The program exits instead of continuing with an invalid camera.

4. Infinite loop
while True:

A webcam keeps producing new frames, so we need to keep reading them until the user chooses to stop.

5. Read a frame
ret, frame = cap.read()

This returns two values:

ret → True if a frame was captured successfully, otherwise False
frame → The image captured from the webcam (stored as a NumPy array)

For example:

ret = True
frame = [[[123,45,67], ...]]
6. Check if reading succeeded
if not ret:
    break

If a frame couldn't be captured (camera disconnected, etc.), stop the loop.

7. Display the frame
cv2.imshow("AI Sign Language Translator", frame)

This creates a window and displays the current frame.

8. Wait for a key
cv2.waitKey(1)

This waits 1 millisecond for a key press.

Without it, the window won't refresh properly.

9. Detect the Q key
if cv2.waitKey(1) & 0xFF == ord('q'):

ord('q') converts the character 'q' into its ASCII code.

If the user presses Q, we exit the loop.

10. Release the camera
cap.release()

This frees the webcam so other programs can use it.

11. Close all windows
cv2.destroyAllWindows()

This closes every OpenCV window cleanly.

What happens internally?
Start Program
      │
      ▼
Open Webcam
      │
      ▼
Capture Frame
      │
      ▼
Display Frame
      │
      ▼
Did user press Q?
      │
 ┌────┴─────┐
 │          │
No         Yes
 │          │
 ▼          ▼
Capture   Release Camera
Next Frame
           │
           ▼
      Close Window
Your task
Create webcam.py.
Copy the code above.
Run:
python webcam.py

You should see your webcam feed.

Press Q to close it.

New things you learned
frame = cv2.flip(frame, 1)

This creates a mirror effect, which feels much more natural when interacting with your own hand.

cv2.putText(...)


'''