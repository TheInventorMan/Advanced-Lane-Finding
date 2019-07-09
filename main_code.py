########################################################
# Advanced Lane Finding code blocks                    #
# Pulled from "Advanced Lane Finding Pipeline.ipynb"   #
# Date: July 3, 2019                                   #
# Course: Udacity Self-Driving Car Engineer Nanodegree #
# Description: Finds lanes in a video taken by a       #
#              dashcam and overlays a polygon over the #
#              estimated lane position/shape. Also     #
#              computes turning radius and position in #
#              the lane.                               #
# See Readme for more information.                     #
# Algorithm output is named "overlayed_video.mp4       #
########################################################

#Import all the packages
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from IPython.display import HTML

%matplotlib inline

####### STAGE 1: Camera calibration #######
#Set file paths as variable names
cal_images = glob.glob('camera_cal/calibration*.jpg')

objp = np.zeros((6*9,3), np.float32) #coordinates of corners in real world
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# step through each image and search for chessboard corners
for fname in cal_images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # if found, add object and image points
    if (ret):
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        

print("Images to process: " + str(len(cal_images)))
img_ctr = 1
for fname in cal_images: #go through each image again
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #undistort each image using mappings from ALL images
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    print("Image " + str(img_ctr) + "/" + str(len(cal_images)))
    img_ctr += 1
    
    continue #comment out to see undistorted images

    #This slows everything down quite a bit oops
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    mpimg.imsave('output_images/' + fname[11:], undistorted)
print("done!")

#store calibration matrices
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


####### STAGE 2: Image/Perspective Correction #######
def undistort_img(distorted):
    undistorted = cv2.undistort(distorted, mtx, dist, None, mtx)
    return undistorted

def perspective_warp(input_img, unwarped_pts, warped_pts):
    img_size = (input_img.shape[1], input_img.shape[0])
    M = cv2.getPerspectiveTransform(unwarped_pts, warped_pts) #compute mtx from points
    warped = cv2.warpPerspective(input_img, M, img_size)#apply transform
    return warped

def perspective_unwarp(input_img, warped_pts, unwarped_pts):
    img_size = (input_img.shape[1], input_img.shape[0])
    Minv = cv2.getPerspectiveTransform(warped_pts, unwarped_pts)
    unwarped = cv2.warpPerspective(input_img, Minv, img_size, flags=cv2.INTER_LINEAR) #apply transform
    return unwarped


####### STAGE 3: Sobel/HLS Filtering #######
def sobel_HLS_filters(raw_img, mag_thresh=(30, 255), dir_thresh=(0.1, 0.7), s_thresh=(120, 255), l_thresh=(130, 255), white_thresh=(210, 255)):
    img = np.copy(raw_img)
    
    #convert to HLS color space and keep L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    #Sobel 
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
    
    #compute direction
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    #compute magnitude
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)#scale
    
    sobel_binary =  np.zeros_like(grad_dir)
    mag_cond = (grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1]) # magnitude condition
    dir_cond =  (grad_dir >= dir_thresh[0]) & (grad_dir <= dir_thresh[1]) #direction condition
    sobel_binary[mag_cond & dir_cond] = 1

    #threshold S and L channels
    color_binary = np.zeros_like(s_channel)
    s_cond = ((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) | (l_channel > white_thresh[0]) #special case for white lines (low S, but *very high* L)
    l_cond = (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])
    color_binary[s_cond & l_cond] = 1
    
    #stack color and sobel filtered channels
    stacked_binary = np.dstack(( np.zeros_like(sobel_binary), color_binary, sobel_binary))# * 255
    return stacked_binary


##For testing, disregard
image = perspective_warp(mpimg.imread('test_images/test2.jpg'), unwarped_pts= np.float32(
    [[585, 460],
    [203, 720],
    [1127, 720],
    [695, 460]]), 
    warped_pts= np.float32(
    [[320, 0],
    [320, 720],
    [960, 720],
    [960, 0]]))

new_img = sobel_HLS_filters(image)

f, ax1 = plt.subplots(1, 1, figsize=(24, 9))
f.tight_layout()
ax1.imshow(new_img)
ax1.set_title('Detected Pixels:', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

####### Stage 4b: Smoothing Lane Jitter #######
leftx_stack = np.ndarray([], dtype=np.int32)
lefty_stack = np.ndarray([], dtype=np.int32)
rightx_stack = np.ndarray([], dtype=np.int32)
righty_stack = np.ndarray([], dtype=np.int32)
left_ctr=np.array([], dtype=np.int32)
right_ctr=np.array([], dtype=np.int32)

def resetSmootherStacks(): #Stacks of activated pixels to smooth over; must be reset at beginning of each video
    #global leftx_stack, lefty_stack, rightx_stack, righty_stack, left_ctr, right_ctr
    leftx_stack = np.ndarray([], dtype=np.int32)
    lefty_stack = np.ndarray([], dtype=np.int32)
    rightx_stack = np.ndarray([], dtype=np.int32)
    righty_stack = np.ndarray([], dtype=np.int32)
    left_ctr=np.array([], dtype=np.int32)
    right_ctr=np.array([], dtype=np.int32)

def smoother(leftx, lefty, rightx, righty): #takes points and pushes onto a stack. Returns last three inputs.
    global leftx_stack, lefty_stack, rightx_stack, righty_stack, left_ctr, right_ctr
    #add to stack
    leftx_stack = np.append(leftx_stack, leftx)
    lefty_stack = np.append(lefty_stack, lefty)
    rightx_stack = np.append(rightx_stack, rightx)
    righty_stack = np.append(righty_stack, righty)
    
    #keep track of number of points added at each call
    left_ctr = np.append(left_ctr, len(leftx))
    right_ctr = np.append(right_ctr, len(rightx))
    
    if len(left_ctr)>3: #only store inputs from last 3 calls
        leftx_stack = np.delete(leftx_stack, np.s_[0:int(left_ctr[0])])
        lefty_stack = np.delete(lefty_stack, np.s_[0:int(left_ctr[0])])
        left_ctr = np.delete(left_ctr, 0)
    if len(right_ctr)>3:
        rightx_stack = np.delete(rightx_stack, np.s_[0:int(right_ctr[0])])
        righty_stack = np.delete(righty_stack, np.s_[0:int(right_ctr[0])])
        right_ctr = np.delete(right_ctr, 0)
    #return whole stack
    return leftx_stack.astype(int), lefty_stack.astype(int), rightx_stack.astype(int), righty_stack.astype(int)    

####### Stage 4a: Main Lane Finding Algorithm #######
left_fit = None #declare in global scope to fix bug
right_fit = None

def find_lane(binary_warped):
    global left_fit, right_fit, left_ctr
    
    margin = 100 #margin for searching around last polynomials
    
    #get detected pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #search around last polynomial starting from 3rd frame
    if len(left_ctr)>3:
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    else:
        #sliding window parameters
        nwindows = 9
        minpix = 100
        window_height = np.int(binary_warped.shape[0]//nwindows)
        
        #flatten color map of detected lines and get histogram of bottom half
        flattened = np.sum(binary_warped, axis=2)
        histogram = np.sum(flattened[flattened.shape[0]//2:,:], axis=0)

        out_img = binary_warped
        #starting point of lane lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            #compute window bounds
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            #get activated pixels in each window
            in_left_window = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            in_right_window = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(in_left_window)
            right_lane_inds.append(in_right_window)

            #recenter window if pixel number threshold met
            if len(in_left_window) > minpix:
                leftx_current = np.int(np.mean(nonzerox[in_left_window]))
            if len(in_right_window) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[in_right_window]))

        try: #hack-y fix
            #only append points in windows
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except:
            pass
    
    #get coordinates of all activated pixels
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    #y values to plot for 
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    #smooth out point detection over last 3 frames
    leftx, lefty, rightx, righty = smoother(leftx, lefty, rightx, righty)
    try:
        #fit polynomial to points
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except:
        #just in case a line was not found
        print("derp")
        left_fit = None
        right_fit = None
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    #mark activated points with colors (red--> left, blue-->right)
    #print(lefty.shape)
    out_img[lefty, leftx] = [1, 0, 0]
    out_img[righty, rightx] = [0, 0, 1]
    
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    
    #make array of all points to draw polygon later
    poly_left = np.dstack((left_fitx, ploty))
    poly_right = np.dstack((right_fitx, ploty))
    poly_pts = np.array(np.concatenate((poly_left, np.flip(poly_right, axis = 1)), axis=0).reshape((-1,1,2)), np.int32)
    
    y_eval = np.max(ploty) #bottom of image
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #compute new polynomial based on points scaled to real world
    left_fit_true = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_true = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    #calculate curvature with true polynomials
    left_curverad = ((1 + (2*left_fit_true[0]*y_eval*ym_per_pix + left_fit_true[1])**2)**1.5) / np.absolute(2*left_fit_true[0])
    right_curverad = ((1 + (2*right_fit_true[0]*y_eval*ym_per_pix + right_fit_true[1])**2)**1.5) / np.absolute(2*right_fit_true[0])    
    
    #car position in lane
    car_pos_px = midpoint - (rightx_base+leftx_base)/2 
    car_pos = round(car_pos_px*xm_per_pix, 2)
    
    return out_img, left_curverad, right_curverad, poly_pts, car_pos

####### Main Pipeline #######
#Constants
unwarped_pts = np.float32(
    [[585, 460],
    [203, 720],
    [1127, 720],
    [695, 460]])

warped_pts = np.float32(
    [[320, 0],
    [320, 720],
    [960, 720],
    [960, 0]])

def main_pipeline(image):
    #1. undistort image
    undistorted = undistort_img(image)
    img_size = image.shape
    
    #2. warp using perspective transform
    warped = perspective_warp(undistorted, unwarped_pts, warped_pts)
    
    #3. apply color and sobel filters to find lane pixels
    filtered_binary = sobel_HLS_filters(warped)
    
    #4. find lanes and compute curve radius and car position in lane
    detected_lane_img, right_curve_rad, left_curve_rad, poly_pts, car_pos_m = find_lane(filtered_binary)
    
    #5. draw lane overlay using polygon points from step 4
    cv2.fillConvexPoly(detected_lane_img, poly_pts,(72,124,150))

    #6. unwarp overlay containing lines and lane
    unwarped_overlay = perspective_unwarp(detected_lane_img, warped_pts, unwarped_pts)
    
    #7. superimpose overlay over initial image to obtain composite image
    unwarped2 = unwarped_overlay.astype(dtype=np.uint8)
    composite = cv2.addWeighted(undistorted, 1., unwarped2, 1., 0.)
    
    #8. make labels
    Rleft = "Left line curvature: " + str(int(left_curve_rad)) + "m"
    Rright = "Right line curvature: " + str(int(right_curve_rad)) + "m"
    
    #sanity check for straight lanes (radius approaches inf, cap at 10km)
    if left_curve_rad > 10000:
        Rleft = "Left line curvature: >10km"
    if right_curve_rad > 10000:
        Rright = "Right line curvature: >10km"
    
    if car_pos_m <= 0:
        carPos = "Car lane position: left " + str(abs(car_pos_m)) + "m" 
    else:
        carPos = "Car lane position: right " + str(abs(car_pos_m)) + "m" 
    
    #9. draw labels on composite image
    cv2.putText(composite, Rleft, (100, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
    cv2.putText(composite, Rright, (700, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
    cv2.putText(composite, carPos, (400, 145), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
    
    return composite

####### Rendering Project Video #######
#testing
image = mpimg.imread('test_images/test2.jpg')
new_img = main_pipeline(image)
resetSmootherStacks()

#don't do both at the same time--breaks the smoothing algo for some reason
showPlot = False
showVideo = True

#if an error is thrown, keep rerunning the cell until it works; there's some sort of dumb race condition in the kernel
if(showPlot):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(new_img)
    ax2.set_title('Final Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

if(showVideo):
    clip1 = VideoFileClip('project_video.mp4')#.subclip(0,5)
    clip_output = 'overlayed_video.mp4'
    clip = clip1.fl_image(main_pipeline)
    %time clip.write_videofile(clip_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(clip_output))
