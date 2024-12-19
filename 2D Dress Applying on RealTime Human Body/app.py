from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import numpy as np
import random
from datetime import datetime
import uuid
import json
from flask_cors import CORS
import numpy as np
import cv2                             
from math import floor
import os
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "static/upload/"
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

def change_shirt_color(img, color):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
   
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 50])
    
    # Create a mask for the shirt
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Convert the user input to the desired BGR color
    color_dict = {
        'blue': [255, 0, 0],
        'white': [255, 255, 255],
        'green': [0, 255, 0],
        'purple': [128, 0, 128]
    }
    bgr_color = np.array(color_dict.get(color, [255, 255, 255]))

    # Apply the new color to the shirt
    img[mask > 0] = bgr_color
    
    return img


cap = None 
def generate_frames(top_image, pant_image):
    global cap

    print(f"Top Image Path: {top_image}")
    print(f"Pants Image Path: {pant_image}")
    cap = cv2.VideoCapture(0)  # Start capturing video when the function is called
    while True:
        imgshirt = cv2.imread(top_image, 1)
        imgpant = cv2.imread(pant_image, 1)

        if imgshirt is None or imgpant is None:
            print("Error loading images.")
            break
        
        shirtgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)
        ret, orig_masks_inv = cv2.threshold(shirtgray, 200, 255, cv2.THRESH_BINARY)
        orig_masks = cv2.bitwise_not(orig_masks_inv)

        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        origpantHeight, origpantWidth = imgpant.shape[:2]  # Get pant dimensions

        pantgray = cv2.cvtColor(imgpant, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
        ret, orig_mask = cv2.threshold(pantgray, 100, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(orig_mask)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        ret, img = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            pantWidth = int(3 * w)
            pantHeight = int(pantWidth * origpantHeight / origpantWidth)  # Use correct pant dimensions

           
            if pantWidth <= 0 or pantHeight <= 0:
                print(f"Invalid pant dimensions: Width={pantWidth}, Height={pantHeight}")
                continue

            x1 = int(x - w // 2)
            x2 = int(x1 + 5 * w // 2)
            y1 = int(y + 5 * h)
            y2 = int(y + h * 14)

            # Ensure coordinates are within image boundaries
            x1, x2 = max(x1, 0), min(x2, width)
            y1, y2 = max(y1, 0), min(y2, height)

            # Resize the pant image
            pant = cv2.resize(imgpant, (pantWidth, pantHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (pantWidth, pantHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (pantWidth, pantHeight), interpolation=cv2.INTER_AREA)

            roi = img[y1:y2, x1:x2]
            if roi.shape[0] != mask_inv.shape[0] or roi.shape[1] != mask_inv.shape[1]:
                mask_inv = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_fg = cv2.bitwise_and(pant, pant, mask=mask)

            if roi_bg.shape != roi_fg.shape:
                print(f"Size mismatch: roi_bg {roi_bg.shape}, roi_fg {roi_fg.shape}")
                continue  

            dst = cv2.add(roi_bg, roi_fg)
            img[y1:y2, x1:x2] = dst

      
            shirtWidth = int(3 * w)
            shirtHeight = int(shirtWidth * origshirtHeight / origshirtWidth)

            if shirtWidth <= 0 or shirtHeight <= 0:
                print(f"Invalid shirt dimensions: Width={shirtWidth}, Height={shirtHeight}")
                continue

            x1s = int(x - w)
            x2s = int(x1s + 3 * w)
            y1s = int(y + h)
            y2s = int(y1s + h * 4)

            # Ensure coordinates are within image boundaries
            x1s, x2s = max(x1s, 0), min(x2s, width)
            y1s, y2s = max(y1s, 0), min(y2s, height)

            shirt = cv2.resize(imgshirt, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_masks, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)

            rois = img[y1s:y2s, x1s:x2s]
            if rois.shape[0] != masks_inv.shape[0] or rois.shape[1] != masks_inv.shape[1]:
                masks_inv = cv2.resize(masks_inv, (rois.shape[1], rois.shape[0]), interpolation=cv2.INTER_AREA)

            roi_bgs = cv2.bitwise_and(rois, rois, mask=masks_inv)
            roi_fgs = cv2.bitwise_and(shirt, shirt, mask=mask)

            if roi_bgs.shape != roi_fgs.shape:
                print(f"Size mismatch: roi_bgs {roi_bgs.shape}, roi_fgs {roi_fgs.shape}")
                continue  # Skip if sizes do not match

            dsts = cv2.add(roi_bgs, roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    top_image = request.args.get('top_image')
    pant_image = request.args.get('pant_image')
    return Response(generate_frames(top_image, pant_image), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    global cap
    if cap is None:
        return jsonify({"message": "Camera not started"}), 500

    success, frame = cap.read()
    if success:
        # Save the captured image
        image_filename = f"face_{str(uuid.uuid4())}.jpg"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)  # Use the filename here
        cv2.imwrite(image_path, frame)  # Save the actual frame, not 'face'

        result = {
            'imageFile': image_filename,
        }

        
        return jsonify({
            'message': 'Face Image captured successfully!',
            'result': result
        })
    return jsonify({"message": "Failed to capture image"}), 500

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global cap
    if cap is not None:
        cap.release()
        return jsonify({"message": "Video feed stopped successfully!"})
    return jsonify({"message": "Video feed not active"}), 500







@app.route('/apply', methods=['GET', 'POST'])
def apply():
    if request.method == 'GET':
        top_image = request.args.get('top_image')
        pant_image = request.args.get('pant_image')
        top_image_path = os.path.join('static', 'assets', 'img', top_image)
        pant_image_path = os.path.join('static', 'assets', 'img', pant_image)
    return render_template('myclothes.html',top_image_path=top_image_path,pant_image_path=pant_image_path, status="0")



@app.route('/viewall')
def viewall():
    return render_template('viewall.html')

@app.route('/withPattern', methods=['GET', 'POST'])
def withPattern():
    if request.method == 'GET':
        top_image = request.args.get('top_image')
        pant_image = request.args.get('pant_image')
        top_image_path = os.path.join('static', 'assets', 'img', top_image)
        pant_image_path = os.path.join('static', 'assets', 'img', pant_image)
        return True


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        top_image = request.args.get('top_image')
        pant_image = request.args.get('pant_image')
        print("----------------------")
        print(top_image)
        print(pant_image)
        print("----------------------")
        cv2.waitKey(1)
        cap=cv2.VideoCapture(0)
        ih=3
        i=2
        while True:
            imgarr=["shirt3.png",'shirt2.png','shirt51.jpg','shirt6.png']

            
            imgshirt = cv2.imread(top_image,1) 
            #user_input_color = input("Enter the shirt color (blue, white, green, purple): ").lower()
            #imgshirt = change_shirt_color(imgshirt, user_input_color)
            if ih==3:
                shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
                ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) 
                orig_masks = cv2.bitwise_not(orig_masks_inv)

            else:
                shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
                ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY)
                orig_masks_inv = cv2.bitwise_not(orig_masks)
            origshirtHeight, origshirtWidth = imgshirt.shape[:2]
            imgarr=["pant7.jpg",'pant21.png']
            #i=input("Enter the pant number you want to try")
            imgpant = cv2.imread(pant_image,1)
            imgpant=imgpant[:,:,0:3]
            pantgray = cv2.cvtColor(imgpant,cv2.COLOR_BGR2GRAY) #grayscale conversion
            if i==1:
                ret, orig_mask = cv2.threshold(pantgray,100 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
                orig_mask_inv = cv2.bitwise_not(orig_mask)
            else:
                ret, orig_mask = cv2.threshold(pantgray,50 , 255, cv2.THRESH_BINARY)
                orig_mask_inv = cv2.bitwise_not(orig_mask)
            origpantHeight, origpantWidth = imgpant.shape[:2]
            face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            ret,img=cap.read()
        
            height = img.shape[0]
            width = img.shape[1]
            resizewidth = int(width*3/2)
            resizeheight = int(height*3/2)
            #img = cv2.resize(img[:,:,0:3],(1000,1000), interpolation = cv2.INTER_AREA)
            cv2.namedWindow("img",cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
            cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)
                pantWidth =  3 * w  #approx wrt face width
                pantHeight = pantWidth * origpantHeight / origpantWidth #preserving aspect ratio of original image..

                # Center the pant..just random calculations..
                if i==1:
                    x1 = x-w
                    x2 =x1+3*w
                    y1 = y+5*h
                    y2 = y+h*10
                elif i==2:
                    x1 = x-w/2
                    x2 =x1+2*w
                    y1 = y+4*h
                    y2 = y+h*9
                else :
                    x1 = x-w/2
                    x2 =x1+5*w/2
                    y1 = y+5*h
                    y2 = y+h*14
                # Check for clipping(whetehr x1 is coming out to be negative or not..)

                #two cases:
                """
                close to camera: image will be to big
                so face ke x+w ke niche hona chahiye warna dont render at all
                """
                if x1 < 0:
                    x1 = 0 #top left ke bahar
                if x2 > img.shape[1]:
                    x2 =img.shape[1] #bottom right ke bahar
                if y2 > img.shape[0] :
                    y2 =img.shape[0] #nichese bahar
                if y1 > img.shape[0] :
                    y1 =img.shape[0] #nichese bahar
                if y1==y2:
                    y1=0
                temp=0
                if y1>y2:
                    temp=y1
                    y1=y2
                    y2=temp
                """
                if y+h > y1: #agar face ka bottom most coordinate pant ke top ke niche hai
                    y1 = 0
                    y2 = 0
                """
                # Re-calculate the width and height of the pant image(to resize the image when it wud be pasted)
                pantWidth = int(abs(x2 - x1))
                pantHeight = int(abs(y2 - y1))
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                # Re-size the original image and the masks to the pant sizes
                """
                if not y1 == 0 and y2 == 0:
                    pant = cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                    mask = cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
                    mask_inv = cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
                # take ROI for pant from background equal to size of pant image
                    roi = img[y1:y2, x1:x2]
                        # roi_bg contains the original image only where the pant is not
                        # in the region that is the size of the pant.
                    num=roi
                    roi_bg = cv2.bitwise_and(roi,num,mask = mask_inv)
                        # roi_fg contains the image of the pant only where the pant is
                    roi_fg = cv2.bitwise_and(pant,pant,mask = mask)
                    # join the roi_bg and roi_fg
                    dst = cv2.add(roi_bg,roi_fg)
                        # place the joined image, saved to dst back over the original image
                    img[y1:y2, x1:x2] = dst
                """
                
                pant = cv2.resize(imgpant, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.resize(orig_mask, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (pantWidth,pantHeight), interpolation = cv2.INTER_AREA)
            # take ROI for pant from background equal to size of pant image
                roi = img[y1:y2, x1:x2]
                    # roi_bg contains the original image only where the pant is not
                    # in the region that is the size of the pant.
                num=roi
                roi_bg = cv2.bitwise_and(roi,num,mask = mask_inv)
                    # roi_fg contains the image of the pant only where the pant is
                roi_fg = cv2.bitwise_and(pant,pant,mask = mask)
                # join the roi_bg and roi_fg
                dst = cv2.add(roi_bg,roi_fg)
                    # place the joined image, saved to dst back over the original image
                top=img[0:y,0:resizewidth]
                bottom=img[y+h:resizeheight,0:resizewidth]
                midleft=img[y:y+h,0:x]
                midright=img[y:y+h,x+w:resizewidth]
                blurvalue=5
                top=cv2.GaussianBlur(top,(blurvalue,blurvalue),0)
                bottom=cv2.GaussianBlur(bottom,(blurvalue,blurvalue),0)
                midright=cv2.GaussianBlur(midright,(blurvalue,blurvalue),0)
                midleft=cv2.GaussianBlur(midleft,(blurvalue,blurvalue),0)
                img[0:y,0:resizewidth]=top
                img[y+h:resizeheight,0:resizewidth]=bottom
                img[y:y+h,0:x]=midleft
                img[y:y+h,x+w:resizewidth]=midright
                img[y1:y2, x1:x2] = dst

        #|||||||||||||||||||||||||||||||SHIRT||||||||||||||||||||||||||||||||||||||||

                shirtWidth =  3 * w  #approx wrt face width
                shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
                # Center the shirt..just random calculations..
                x1s = x-w
                x2s =x1s+3*w
                y1s = y+h
                y2s = y1s+h*4
                # Check for clipping(whetehr x1 is coming out to be negative or not..)

                if x1s < 0:
                    x1s = 0
                if x2s > img.shape[1]:
                    x2s =img.shape[1]
                if y2s > img.shape[0] :
                    y2s =img.shape[0]
                temp=0
                if y1s>y2s:
                    temp=y1s
                    y1s=y2s
                    y2s=temp
                """
                if y+h >=y1s:
                    y1s = 0
                    y2s=0
                """
                # Re-calculate the width and height of the shirt image(to resize the image when it wud be pasted)
                shirtWidth = int(abs(x2s - x1s))
                shirtHeight = int(abs(y2s - y1s))
                y1s = int(y1s)
                y2s = int(y2s)
                x1s = int(x1s)
                x2s = int(x2s)
                """
                if not y1s == 0 and y2s == 0:
                    # Re-size the original image and the masks to the shirt sizes
                    shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                    mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                    masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                    # take ROI for shirt from background equal to size of shirt image
                    rois = img[y1s:y2s, x1s:x2s]
                        # roi_bg contains the original image only where the shirt is not
                        # in the region that is the size of the shirt.
                    num=rois
                    roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
                    # roi_fg contains the image of the shirt only where the shirt is
                    roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
                    # join the roi_bg and roi_fg
                    dsts = cv2.add(roi_bgs,roi_fgs)
                    img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
                """
                # Re-size the original image and the masks to the shirt sizes
                shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
                mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
                # take ROI for shirt from background equal to size of shirt image
                rois = img[y1s:y2s, x1s:x2s]
                    # roi_bg contains the original image only where the shirt is not
                    # in the region that is the size of the shirt.
                num=rois
                roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
                # roi_fg contains the image of the shirt only where the shirt is
                roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
                # join the roi_bg and roi_fg
                dsts = cv2.add(roi_bgs,roi_fgs)
                img[y1s:y2s, x1s:x2s] = dsts # place the joined image, saved to dst back over the original image
                #print "blurring"
                
                break
            cv2.imshow("img",img)
            #cv2.setMouseCallback('img',change_dress)
            if cv2.waitKey(100) == ord('q'):
                break

        cap.release()                           # Destroys the cap object
        cv2.destroyAllWindows()                 # Destroys all the windows created by imshow

    return render_template('viewall.html')



@app.route('/generate_dress_with_pattern', methods=['GET', 'POST'])
def generate_dress_with_pattern():
    if request.method == 'POST':
        data = request.get_json()  # Get JSON data from the request
        top_image = data.get('top_image')  # Extract the top image name
        bottom_image_name = data.get('bottom_image')  # Extract the bottom image name
        pattern_image = data.get('pattern_image')  # Extract the pattern image name
        generate_image_1(top_image,pattern_image)
        generate_image1(top_image,pattern_image)
        print(bottom_image_name)  # Debugging output
        print(pattern_image)  # Debugging output

        # Prepare a response dictionary
        response_data = {
            'top_image_path': top_image,
            'bottom_image_path': bottom_image_name,
            'pattern_image_path': pattern_image,
            'status': "1"  
        }

        # Respond with JSON
        return jsonify(response_data)

    return jsonify({"error": "Method Not Allowed"}), 405  # Handle non-POST requests


def generate_image_1(blouse_image_path, flower_pattern_path):
    # Open the blouse and flower pattern images
    blouse_image = Image.open(f'static/assets/img/{blouse_image_path}').convert("RGBA")
    flower_pattern = Image.open(f'static/assets/img/{flower_pattern_path}').convert("RGBA")

    # Convert flower pattern to a numpy array
    data = np.array(flower_pattern)

    # Identify black pixels and make them transparent
    black_pixels = (data[:, :, :3] == [0, 0, 0]).all(axis=-1)
    data[black_pixels, 3] = 0

    # Create a new image with no background
    flower_pattern_no_bg = Image.fromarray(data)

    # Resize the flower pattern to better fit the blouse
    flower_pattern_resized = flower_pattern_no_bg.resize((blouse_image.width // 2, blouse_image.height // 2))

    # Paste the resized flower pattern onto the blouse image
    blouse_with_pattern = blouse_image.copy()
    blouse_with_pattern.paste(flower_pattern_resized, (blouse_image.width // 4, blouse_image.height // 4), flower_pattern_resized)

    # Convert the PIL image to a numpy array for OpenCV
    blouse_with_pattern_cv = cv2.cvtColor(np.array(blouse_with_pattern), cv2.COLOR_RGBA2BGR)

    # Define the output path and save the image using OpenCV
    image_filename = 'g1.png'
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
    
    # Make sure the output directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Save the image
    cv2.imwrite(image_path, blouse_with_pattern_cv)

    return image_path  # Optionally return the path to the saved image if needed

def generate_image1(blouse_image_path, flower_pattern_path):
    blouse_image = Image.open(f'static/assets/img/{blouse_image_path}').convert("RGBA")
    flower_pattern = Image.open(f'static/assets/img/{flower_pattern_path}').convert("RGBA")

    # Convert the flower pattern to a numpy array
    data = np.array(flower_pattern)

    # Identify black pixels and make them transparent
    black_pixels = (data[:, :, :3] == [0, 0, 0]).all(axis=-1)
    data[black_pixels, 3] = 0
    flower_pattern_no_bg = Image.fromarray(data)

    # Resize the flower pattern
    flower_pattern_resized = flower_pattern_no_bg.resize((blouse_image.width // 2, blouse_image.height // 2))

    def blend_image_with_pattern(blouse_image, flower_to_apply, position, output_filename, alpha=0.5):
        blouse_copy = blouse_image.copy()
        blouse_copy.paste(flower_to_apply, position, flower_to_apply)
        
        # Blend the images
        blended_image = Image.blend(blouse_image, blouse_copy, alpha)

        # Save the blended image using OpenCV
        image_filename = output_filename
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)  
        
        # Convert the blended image to a numpy array
        blended_image_cv = cv2.cvtColor(np.array(blended_image), cv2.COLOR_RGBA2BGR)

        # Ensure the output directory exists
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        # Save the image using OpenCV
        cv2.imwrite(image_path, blended_image_cv)
        print(f"Image saved as {image_path}")

    # Full pattern application
    blend_image_with_pattern(
        blouse_image,
        flower_pattern_resized,
        (blouse_image.width // 4, blouse_image.height // 4),
        'g2.png',
        alpha=0.5
    )

    # Partial pattern application (cropped)
    flower_cropped = flower_pattern_resized.crop((0, 0, flower_pattern_resized.width // 2, flower_pattern_resized.height))
    blend_image_with_pattern(
        blouse_image,
        flower_cropped,
        (blouse_image.width // 4, blouse_image.height // 4),
        'g3.png',
        alpha=0.5
    )

    # Random pattern application
    max_x = blouse_image.width - flower_pattern_resized.width
    max_y = blouse_image.height - flower_pattern_resized.height
    random_position = (
        random.randint(0, max_x),
        random.randint(0, max_y)
    )
    blend_image_with_pattern(
        blouse_image,
        flower_pattern_resized,
        random_position,
        'g4.png',
        alpha=0.5
    )
    
    print("All blended images generated successfully.")


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)