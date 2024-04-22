from flask import Flask, render_template, request, redirect, url_for, flash, Response
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

import imutils
import depthai as dai
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

app = Flask(__name__)
app.secret_key = 'supersecretkey'

def convert_milli_to_inch(x):
    return x / 254

# find circle properties
def find_circle_properties(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=250)

    if circles is not None:
        circles = circles[0, :]
        x, y, r = circles[0]
        center = (int(x), int(y))
        width = int(2 * r)
        height = int(2 * r)
        return center, width, height
    else:
        print("No circle detected in the image.")
        return None, None, None

# Function to find circular object diameter (replace with actual logic)
def find_diameter(img_path, object_dist = 300):
    # Mock logic to find diameter
    camera_matrix = np.loadtxt('static/data/camera_matrix.txt')
    # Read image
    image = cv2.imread(img_path)
    
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    Z = object_dist

    (x, y), w, h = find_circle_properties(image)
    bbox = (x, y, w, h)
    
    x, y, w, h = bbox

    Image_point1x = x
    Image_point1y = y
    Image_point2x = x + w
    Image_point2y = y + h

    cv2.line(image, (Image_point1x, Image_point1y-h//2), (Image_point1x, Image_point2y-h//2), (0, 255, 0), 8)

    Real_point1x = Z * (Image_point1x / fx)
    Real_point1y = Z * (Image_point1y / fy)
    Real_point2x = Z * (Image_point2x / fx)
    Real_point2y = Z * (Image_point2x / fy)

    dist = math.sqrt((Real_point2y - Real_point1y) ** 2 + (Real_point2x - Real_point1x) ** 2)
    val = round(convert_milli_to_inch(dist) * 10, 2)

    cv2.putText(image, str(val) + " cm", (Image_point1x - 200, (y + h) // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_img_path = "static/results/circlular_objects/"+img_path.split("/")[-1]
    cv2.imwrite(out_img_path, image)

    return val, out_img_path

# Define function to compute integral image
def compute_integral_image(frame):
    gray_clr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    integral_image = np.cumsum(np.cumsum(gray_clr, axis=1), axis=0)
    return integral_image

# Define function to stitch images
def stitch_images(img1, img2):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(gray_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    good_matches = np.asarray(good_matches)
    
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
        
        stitched_img = warped_img1.copy()
        stitched_img[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        return stitched_img
    else:
        raise AssertionError("Insufficient good matches for stitching.")
    
########## Object Tracking #######

def detect_qr(frame):
    qcd = cv2.QRCodeDetector()
    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
    if ret_qr:
        for s, p in zip(decoded_info, points):
            if s:
                frame = cv2.polylines(frame, [p.astype(int)], True, (255, 255, 255), 8)
                cv2.putText(frame, s, (int(p[0][0]), int(p[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                color = (0, 0, 0)
                frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
    return frame

class ObjTracker:
    def __init__(self):
        self.obj_centers = {}
        self.obj_id_count = 0

    def update(self, frame):
        roi = frame
        obj_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        mask = obj_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        tracked_boxes_ids = self._track_objs(detections)
        for box_id in tracked_boxes_ids:
            x, y, w, h, obj_id = box_id
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return roi

    def _track_objs(self, obj_rectangles):
        tracked_objects = []
        for rect in obj_rectangles:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            obj_detected = False
            for obj_id, center in self.obj_centers.items():
                dist = math.hypot(cx - center[0], cy - center[1])
                if dist < 25:
                    self.obj_centers[obj_id] = (cx, cy)
                    tracked_objects.append([x, y, w, h, obj_id])
                    obj_detected = True
                    break
            if not obj_detected:
                self.obj_centers[self.obj_id_count] = (cx, cy)
                tracked_objects.append([x, y, w, h, self.obj_id_count])
                self.obj_id_count += 1

        new_obj_centers = {}
        for obj_bb_id in tracked_objects:
            _, _, _, _, obj_id = obj_bb_id
            center = self.obj_centers[obj_id]
            new_obj_centers[obj_id] = center

        self.obj_centers = new_obj_centers.copy()
        return tracked_objects

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

class ObjDimensionMarker:
    def __init__(self, obj_width_inches):
        self.obj_width_inches = obj_width_inches
        self.pixelsPerMetric = None

    def mark_obj_dims(self, gray_frame):
        gray = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        result_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(result_frame, [box.astype("int")], -1, (0, 0, 0), 2)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(result_frame, (int(tltrX), int(tltrY)), 5, (0, 0, 0), -1)
            cv2.circle(result_frame, (int(blbrX), int(blbrY)), 5, (0, 0, 0), -1)
            cv2.circle(result_frame, (int(tlblX), int(tlblY)), 5, (0, 0, 0), -1)
            cv2.circle(result_frame, (int(trbrX), int(trbrY)), 5, (0, 0, 0), -1)

            cv2.line(result_frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 2)
            cv2.line(result_frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if self.pixelsPerMetric is None:
                self.pixelsPerMetric = dB / self.obj_width_inches

            dimA_cm = dA / self.pixelsPerMetric * 2.54
            dimB_cm = dB / self.pixelsPerMetric * 2.54

            cv2.putText(result_frame, "{:.1f}in".format(dimA_cm), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(result_frame, "{:.1f}in".format(dimB_cm), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return result_frame

##################################

# Function to validate uploaded file
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_dimensions', methods=['POST'])
def real_dimensions():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        diameter, out_img_path = find_diameter(file_path)
        result_msg = f"Diameter of circular object is {diameter} cm."
        return render_template('result.html', filename=out_img_path, result_msg=result_msg)
    else:
        flash('Invalid file type')
        return redirect(request.url)
    
@app.route('/integral_image', methods=['POST'])
def integral_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        frame = image = cv2.imread(file_path)
        integral_img = compute_integral_image(frame)
        out_img_path = "static/results/integral_images/"+file_path.split("/")[-1]
        plt.imsave(out_img_path, integral_img[..., ::-1])
        result_msg = "Integral of given RGB image is:"
        return render_template('result.html', filename=out_img_path, result_msg=result_msg)
    else:
        flash('Invalid file type')
        return redirect(request.url)
    
@app.route('/create_panorama', methods=['POST'])
def create_panorama():
    if 'images' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    img_files = request.files.getlist('images')
    if img_files:
        img_paths = []
        for img_file in img_files:
            img_path = os.path.join('static/uploads', img_file.filename)
            img_file.save(img_path)
            img_paths.append(img_path)

        images = [cv2.imread(img_path) for img_path in img_paths]

        panorama = images[0]
        for image in images[1:]:
            panorama = stitch_images(panorama, image)
        
        out_img_path = "static/results/panorama_imgs/panorama_output.png"
        cv2.imwrite(out_img_path, panorama)
        result_msg = "Panorama after stitching given set of images is:"
        return render_template('result.html', filename=out_img_path, result_msg=result_msg)
    else:
        flash('Invalid file type')
        return redirect(request.url)
    
def process_video_stream():
    obj_width = 0.995
    try:
        # Create object tracker instance
        tracker = ObjTracker()

        # Initialize object dimension marker
        object_marker = ObjDimensionMarker(obj_width_inches=obj_width)

        # Create a pipeline
        pipeline = dai.Pipeline()

        # Define a node for the stereo camera
        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)

        # Define a node for the left camera
        left = pipeline.createMonoCamera()
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Define a node for the right camera
        right = pipeline.createMonoCamera()
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Connect left and right camera outputs to the stereo node
        left.out.link(stereo.left)
        right.out.link(stereo.right)

        # Define a node for the output
        xoutDepth = pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # Link stereo camera output to the output node
        stereo.depth.link(xoutDepth.input)

        # Define a node to get left camera frames
        xoutLeft = pipeline.createXLinkOut()
        xoutLeft.setStreamName("left")

        # Link left camera output to the output node
        left.out.link(xoutLeft.input)

        # Define a node to get right camera frames
        xoutRight = pipeline.createXLinkOut()
        xoutRight.setStreamName("right")

        # Link right camera output to the output node
        right.out.link(xoutRight.input)

        # Define a source - color camera
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # Create RGB output
        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.video.link(xout.input)

        # Connect to the device
        with dai.Device(pipeline) as device:
            # Output queues
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            leftQueue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
            rightQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            # Start the pipeline
            device.startPipeline()

            # OpenCV setup
            camera_id = 0
            delay = 1
            window_name = 'Object Tracking & Recgnition through QR Code'
            cap = cv2.VideoCapture(camera_id)

            while True:
                # Get the depth frame
                inDepth = depthQueue.get()

                # Get the left camera frame
                inLeft = leftQueue.get()

                # Get the right camera frame
                inRight = rightQueue.get()

                # Get the rgb camera frame
                inSrc = rgbQueue.get()

                # Access the depth data
                depthFrame = inDepth.getFrame()

                # Access the left camera frame
                leftFrame = inLeft.getCvFrame()

                # Access the right camera frame
                rightFrame = inRight.getCvFrame()
                
                # Data is originally represented as a flat 1D array, it needs to be converted into HxW form
                rgbFrame = inSrc.getCvFrame()

                # Combine left and right frames horizontally
                stereoFrame = cv2.hconcat([leftFrame, rightFrame])

                # Detect QR codes in the stereo frame
                stereoFrame = detect_qr(stereoFrame)

                # Perform object detection and tracking
                stereoFrame = tracker.update(stereoFrame)

                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2GRAY)

                # Mark object dimensions
                frame = object_marker.mark_obj_dims(stereoFrame)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except:
        # Create object tracker instance
        tracker = ObjTracker()

        # Initialize object dimension marker
        object_marker = ObjDimensionMarker(obj_width_inches=obj_width)

        cap = cv2.VideoCapture(0)  # Use the correct device index
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect QR codes in the stereo frame
            frame = detect_qr(frame)

            # Perform object detection and tracking
            frame = tracker.update(frame)

            # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = object_marker.mark_obj_dims(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
@app.route('/stream_video')
def stream_video():
    return Response(process_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/object_tracking', methods=['GET','POST'])
def object_tracking():
    try:
        return render_template('object_tracking.html')
    except:
        flash('Invalid file type')
        return redirect(request.url)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error='404 - Page not found'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html', error='500 - Internal Server Error'), 500

if __name__ == '__main__':
    app.run(debug=True)
