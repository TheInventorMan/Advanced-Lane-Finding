# Advanced Lane Finding Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./completed/calibrated.jpg "Chessboard"
[image2]: ./completed/undistorted.jpg "Undistorted"
[image3]: ./completed/warped.jpg "Road Transformed"
[image4]: ./completed/thresholded.jpg "Thresholded Image"
[image5]: ./completed/detected_lines.jpg "Fit Visual"
[image6]: ./completed/final.jpg "Output"
[video1]: ./completed/overlayed_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced Lane Finding Pipeline.ipynb"

Calibrating the camera is necessary to be able to reliably obtain real-world measurements from the image. To do this, I needed to set up an array to store all of the detected chessboard corners, as well as the real-world coordinates in space. I used some of the example code to initialize this array. Next, each image is converted to grayscale, before being passed to the `findChessboardCorners()` function. If the corners are found, their coordinates, along with the corresponding real-world coordinates, are appended to their respective arrays. After that, all of the images are processed one-by-one, using the two arrays to generate distortion and camera matrices, and applying the required transformations to the image. 

The correction matrices are stored for undistorting the pipeline images later. 

Undistorted chessboard images can be found in the output_images/ directory, and an example is shown below:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

From the camera calibration step, I stored the camera and distortion matrices in global variables, which were then used with the `cv2.undistort()` function to undistort each image as shown below:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I wrote a few helper functions to interface with OpenCV's built in undistort and perspective transform functions. Those functions take in source and destination points that represent a mapping from the unwarped to the warped image. The points are given in lines 2-12 in the "Full Pipeline" section.

One thing that I have done differently in my implementation, is that I performed the perspective warp *before* the color filtering, so that I did not have to worry too much about the surroundings when tuning the filter parameters for line detection.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Once I had the transformed top-down view image of the lanes, I passed it on to the sobel/HLS filter to detect the lane lines. I applied both magnitude and direction thresholds for the sobel filter, and both saturation and lightness thresholds for the HLS filter. The sobel filter used a kernel size of 5 to eliminate some noise.

The HLS filter had a special case for the white lines, where if the lightness is sufficiently high, the saturation is not considered. Meanwhile, the yellow line is highly saturated, though not *extremely* high on the lightness value.

The filtered image is shown below, with the color filter yielding green pixels, and the sobel filter yielding blue pixels:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The filtered image was then sent to the lane detection function, which took in the activated pixels and passed them on to the `cv2.polyfit()` function to fit two quadratic lines onto the image for the left and right lanes.

Eliminating noise from the filtered image was a challenge. To overcome it, I first took the histogram of the lower half of the image to find the peaks in number of activated pixels along the x-axis. These peaks were used as the starting x-coordinates for each of the two lines. Then, a window with a height of 1/9th the frame height is drawn 100 pixels to the left and right of this starting point. If there are more than 100 pixels in this window, it is recentered around the mean of these points. I would then iterate upwards, repeat this process for each subsequent window, rejecting points outside the window to the left and right.

Once the initial polynomial functions are found, I only searched near the polynomial in each subsequent frame. This strategy sped up the search process, as well as became more robust against outliers and noise. 

I also implemented a function to smooth out the lane line transitions between frames. This is achieved by maintaining a stack of activated in-window points that stores the points from the last three frames, so that the `cv2.polyfit()` function can use all of those points to fit the polynomial. Simply averaging the polynomial coefficients would have yielded unexpected results, so I took the other approach instead. 

The image with the detected lines plotted is shown below: (Left lane pixels are colored red, and right lane pixels are colored blue)

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To find the radius of curvature, I first scaled each of the activated points into the world frame, by multiplying the x-coordinates by 3.7/700 and the y-coordinates by 30/720. Next, the `cv2.polyfit()` function was called again, to get the polynomial coefficients corresponding to the function in real-world coordinates. Finally, the radius of curvature equation was used on this new polynomial function to determine the real-world radius of curvature. The code that performs this is in lines 118-123 of the `find_lane()` function.

To find the relative position of the car in the lane, I used the histogram peaks that were detected and took their average, to find the relative position of the lane to the car. Subtracting this number from the midpoint of the image and multiplying by 3.7/700 yields the relative position of the car in the lane, in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The overlay containing the detected lines and lane polygon is then transformed and overlayed over the input image in lines 32-36 in the `main_pipeline()` function, producing the following result:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./completed/overlayed_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Perhaps the most difficult part was tuning the color and Sobel filter parameters. I was attempting to only use Sobel_x magnitude and Saturation for the threshold parameters, but I later expanded it to also consider Lightness, absolute Sobel magnitude, and Sobel direction, and the results are quite promising. However, it did not fare too well with the challenge videos since the parameters would need more tuning, but this would be a future improvement. It seems to fail when there is a vertical asphalt line in the pavement, which causes a lot of Sobel noise in those areas. Also, the pipeline seems to fail with faded lane lines, which do not have a high enough Saturation or Lightness to be detected.

One key difference with my implementation was that I chose to do the perspective transform before applying the color and Sobel filters. This way, I would not have to worry too much about the surroundings while tuning the parameters.

Additionally, I implemented a smoothing algorithm to reduce jitter in the lane detection. I did this by passing all of the points from the last three frames to the `cv2.polyfit()` function, so that it can take a quasi-average over the three frames.

This may be a long shot, but a potential improvement could be to vary the thresholds based on a color histogram of the entire image. The histogram could be used to determine overall lighting conditions and adjusts the thresholds accordingly.
