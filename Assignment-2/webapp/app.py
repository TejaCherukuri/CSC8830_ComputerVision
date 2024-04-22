from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

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

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error='404 - Page not found'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html', error='500 - Internal Server Error'), 500

if __name__ == '__main__':
    app.run(debug=True)
