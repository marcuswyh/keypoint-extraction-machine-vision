import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def task1():
    #============
    # SUBTASK A
    #============
    # define path
    pre = os.path.dirname(os.path.realpath(__file__))
    fname = "Assignment_MV_01_image_1.jpg"
    path = os.path.join(pre, fname)

    # read img
    img = cv2.imread(path)
    # convert img to grayscale and convert to float 32
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    # get dimensions of img
    height, width = img.shape[:]
    # resize and double the size of img
    img_resize = cv2.resize(img, (width*2,height*2))


    #============
    # SUBTASK B
    #============
    # method to obtain sigma value
    def sigma(k):
        return 2**(k/2)

    # initialize xy values based on image size
    x,y = np.meshgrid(np.arange(0,len(img[0])),np.arange(0,len(img)))
    taskB_arr = []
    # loop through sigma sizes
    for i in range(12):
        # calculate gaussian smoothing kernels of certain sigma sizes
        gaussian_smooth_kernel = np.exp(-((x-len(img[0])/2)**2+(y-len(img)/2)**2)/(2*sigma(i)**2))/(2*np.pi*sigma(i)**2)

        # display kernels as image with windows size of +-3 sigma
        midpoint = (gaussian_smooth_kernel.shape[0]/2,gaussian_smooth_kernel.shape[1]/2)
        plt.imshow(gaussian_smooth_kernel[int(midpoint[0]-(3*sigma(i))):int(midpoint[0]+(3*sigma(i))), int(midpoint[1]-(3*sigma(i))):int(midpoint[1]+(3*sigma(i)))])
        plt.show()
        

        # apply kernels on resized image
        res = cv2.filter2D(img_resize, -1, gaussian_smooth_kernel)
        taskB_arr.append([res,i,sigma(i)])

    # display resulting images with kernels applied
    for res in taskB_arr:
        # convert image to uint8 for display purposes
        applied_kernel = res[0].astype(np.uint8)
        cv2.namedWindow("Gaussian with sigma size {}".format(res[1]), cv2.WINDOW_NORMAL)
        cv2.imshow("Gaussian with sigma size {}".format(res[1]), applied_kernel)
        cv2.waitKey()
        cv2.destroyAllWindows()


    #============
    # SUBTASK C
    #============
    taskC_arr = []
    for i in range(len(taskB_arr)):
        if i < len(taskB_arr)-1:
            # subtract image of larger sigma with image of smaller sigma to obtain DoG image
            diff = (taskB_arr[i+1])[0] - (taskB_arr[i])[0]

            # display resulting DoG image
            dog = diff.astype(np.uint8)
            cv2.namedWindow("DoG {}".format(i), cv2.WINDOW_NORMAL)
            cv2.imshow("DoG {}".format(i), dog)
            cv2.waitKey()
            cv2.destroyAllWindows()

            # append DoG and associated sigma value
            taskC_arr.append((diff, (taskB_arr[i+1])[2]))

    #============
    # SUBTASK D
    #============
    def non_maximum_suppression(image,T):
        # obtain 3 layers of image data
        img_arr, sigma_arr = [], []
        for img in image:
            img_arr.append(img[0])
            sigma_arr.append(img[1])

        points = []
        # for each pixel coordinate
        for x in range(1,len((img_arr)[0])-1):
            for y in range(1,len(((img_arr)[0])[0])-1):

                # if current coordinate meets threshold value
                if (((img_arr[1])[x,y]>T) and
                    # 8 neighbors of current coordinate's image
                    ((img_arr[1])[x,y]>(img_arr[1])[x-1,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x-1,y]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x-1,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x+1,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x+1,y]) and
                    ((img_arr[1])[x,y]>(img_arr[1])[x+1,y+1]) and
                    
                    # 9 neighbors of previous layer's image pixel coordinates
                    ((img_arr[1])[x,y]>(img_arr[0])[x-1,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x-1,y]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x-1,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x+1,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x+1,y]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x+1,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[0])[x,y]) and

                    # 9 neighbors of next layer's image pixel coordinates
                    ((img_arr[1])[x,y]>(img_arr[2])[x-1,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x-1,y]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x-1,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x+1,y-1]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x+1,y]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x+1,y+1]) and
                    ((img_arr[1])[x,y]>(img_arr[2])[x,y])):

                    # return coordinates (x,y), and return sigma in which it was detected
                    points.append([x, y, sigma_arr[1]])

        print ("number of feature points found: ", len(points))
        return points

    t = 10
    taskD_arr = []

    # loop through from 2nd img to second last image to compare in DoG cube form
    for i in range(1,len(taskC_arr)-1):
        # calculate non max suppresion with stacks of 3 images (e.g images[1,2,3], images[2,3,4] etc.)
        res = non_maximum_suppression((taskC_arr[i-1:i+2]), t)
        taskD_arr.append(res)

        print("obtained features for image {}".format(i))


    #============
    # SUBTASK E
    #============

    dx = np.array([[1, 0, -1]])
    dy = np.array([[1],
                  [0],
                  [-1]])

    taskE_arr = []

    for im in taskB_arr:

        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(1,2) 

        derivativex = cv2.filter2D(im[0],-1,dx)
        derivativey = cv2.filter2D(im[0],-1,dy)

        taskE_arr.append([derivativex, derivativey, im[2]])

        # # use the created array to output multiple images
        displayx = derivativex.astype(np.uint8)
        displayy = derivativey.astype(np.uint8)
        axarr[0].imshow(displayx)
        axarr[1].imshow(displayy)
        plt.show()


    #============
    # SUBTASK F
    #============
    taskF_arr = []

    def gradientMagnitude(dx,dy):
        # returns calculated gradient magnitude
        return np.sqrt((dx**2) + (dy**2))

    def gradientDirection(dx, dy):
        # calculate direction radiant
        # convert radiant to degrees
        return np.arctan2(dy,dx)*360/(2*np.pi)

    def weightingFunc(matrix, sigma):
        # if matrix is not empty and size is 7x7
        if np.size(matrix,1) > 0 and np.size(matrix,0) > 0 and np.size(matrix,1) == 7 and np.size(matrix,0) == 7:
            for x in range(len(matrix)):
                for y in range(len(matrix)):
                    # apply and assign weights to each of cell in 7x7 grid
                    weight = np.exp(-(x**2 + y**2) / (9*(sigma**2))) / ((9 * np.pi * (sigma**2)) / 2)
                    matrix[x,y] = weight
        return matrix

    def histogram(mag, dir, bins):
        size = dir.shape[0]
        histogram = np.zeros(shape=(bins.size))

        # loop through cells
        for x in range(size):
            for y in range(size):
                # current direction and magnitude of cell
                direction = dir[x,y]
                magnitude = mag[x,y]

                # increment value of corresponding histogram bin based on direction and magnitude
                for idx,bin in enumerate(bins):
                    if np.around(direction, decimals=-1) == bin:
                        histogram[idx] +=1
        return histogram

    # 36-bin histogram
    hist_bin = []
    for i in range(1,37):
        hist_bin.append(i*10)
    hist_bin = np.array(hist_bin)

    # temporary image for drawing circles
    temp = cv2.imread(path)
    temp = cv2.resize(temp, (width*2,height*2))

    # array of colors containing rgb tuples
    colors = [(255,0,0), (255,127,0), (255,255,0), (0,255,0), (0,0,255), (60,60,110), (139,0,255), (255,255,255), (0,0,0)]

    # key points list of each image (9 images)
    for i,image in enumerate(taskD_arr):
        # xy derivatives of 12 images
        for derivative in taskE_arr:
            # if keypoint sigma matches the derivatives' sigma
            if (image[0])[2] == derivative[2]:
                # sigma value
                sigma = derivative[2]

                # calculate gradient magnitude and direction
                grad_magnitude = gradientMagnitude(derivative[0],derivative[1])
                grad_direction = gradientDirection(derivative[0],derivative[1])

                # temporary image for drawing circles
                temp_img = temp.copy()

                # loop through keypoints in image
                x,y=0,0
                for keypoint in image:
                    # obtain xy coordinates of keypoint
                    x,y = keypoint[0], keypoint[1]
                    # apply weighting function to magnitude of 7x7 grid from the keypoint
                    magnitude = weightingFunc(grad_magnitude[x-3:x+4, y-3:y+4], sigma)
                    # obtain 7x7 grid of coordinates from keypoints on gradient direction
                    direction = grad_direction[x-3:x+4, y-3:y+4]
                    # convert the negative degree values to encompass 180-360 degrees range
                    direction = (direction + 360) % 360

                    # if magnitude is not empty
                    if magnitude.size != 0 and np.size(magnitude,1) == 7 and np.size(magnitude,0) == 7:
                        # get index value of highest histogram value from gradient histogram
                        keypoint_direction_idx = np.argmax(histogram(magnitude, direction, hist_bin))
                        # get direction value using histogram index
                        keypoint_direction = hist_bin[keypoint_direction_idx]

                        # draw circles and directional lines for all keypoints
                        xpoint = int(x + ((3*sigma) * math.cos(np.deg2rad(keypoint_direction))))
                        ypoint = int(y + ((3*sigma) * math.sin(np.deg2rad(keypoint_direction))))
                        cv2.circle(temp, (y,x), int(3*sigma), colors[i], 2)
                        cv2.line(temp, (y,x), (ypoint,xpoint), colors[i], 2)

                print ("processed image ")
                taskF_arr.append(temp.copy())


    #============
    # SUBTASK G
    #============

    for img in taskF_arr:
        cv2.namedWindow("", cv2.WINDOW_NORMAL) 
        cv2.imshow("", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(pre , "task1_final_result.png"), taskF_arr[-1])


def task2():
    #============
    # SUBTASK A
    #============
    pre = os.path.dirname(os.path.realpath(__file__))
    fname = "Assignment_MV_01_image_1.jpg"
    path1 = os.path.join(pre, fname)

    fname = "Assignment_MV_01_image_2.jpg"
    path2 = os.path.join(pre, fname)

    # read img
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # convert img to grayscale and convert to float 32
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)

    # get dimensions of img
    height, width = img1.shape[:]

    
    #============
    # SUBTASK B
    #============
    # define window coordinate
    window_coord = [(360,210), (430,300)]
    temp = cv2.imread(path1)
    # draw rectangle on window coordinate
    cv2.rectangle(temp, window_coord[0], window_coord[1], (0,255,0), 2)

    # display image with drawn rectangle
    cv2.imshow("drawn rectangle", temp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # obtain window cutout
    window_cutout = img1[(window_coord[0])[1]:(window_coord[1])[1], (window_coord[0])[0]:(window_coord[1])[0]]

    # display cutout
    temp = temp[(window_coord[0])[1]:(window_coord[1])[1], (window_coord[0])[0]:(window_coord[1])[0]]
    cv2.imshow("window cutout", temp)
    cv2.waitKey()
    cv2.destroyAllWindows()


    #============
    # SUBTASK C
    #============
    # calculate mean and standard deviation of window cutout
    cutout_mean = np.mean(window_cutout)
    cutout_std = np.std(window_cutout)

    # obtain dimensions of window cutout
    window_size = [(window_coord[1])[0] - (window_coord[0])[0], (window_coord[1])[1] - (window_coord[0])[1]]

    # loop through all pixels with window size
    coord, cross_corrs = [], []
    for y in range(width-(window_size[0]+1)):
        for x in range(height-(window_size[1]+1)):
            # obtain cutout with window dimensions for current coordinate
            input_cutout = img2[x:x+window_size[1],y:y+window_size[0]]
            # calculate mean for current coordinate cutout
            input_mean = np.mean(input_cutout)
            
            # cross correlation formula denominator and numerator
            numerator = np.sum((input_cutout - input_mean) * (window_cutout - cutout_mean))
            denominator = np.sqrt(np.sum((input_cutout - input_mean)**2)) * np.sqrt(np.sum((window_cutout - cutout_mean)**2))

            # calculate cross correlation for each cutout at current coordinate
            cross_correlation = numerator / denominator

            # append current coordinate and calculated cross correlation value
            coord.append((x,y))
            cross_corrs.append(cross_correlation)
    
    # cross corr display
    corr_img = np.zeros(img1.shape)
    for i in range(len(coord)):
        x,y = (coord[i])[0], (coord[i])[1]
        corr_img[x,y] = cross_corrs[i]

    # display cross correlation image
    cv2.imshow("correlation image", corr_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # get index of maximum cross correlation value
    idx = np.argmax(cross_corrs)
    # obtain resulting coordinate of max correlation value
    result_coord = coord[idx]

    print ("Max cross correlation value: ",cross_corrs[idx])
    print ("Resulting coordinate of max correlation value: ",(result_coord[1], result_coord[0]))

    # draw rectangle on input image 2 using max correlation value coordinate
    temp2 = cv2.imread(path2)
    cv2.rectangle(temp2, (result_coord[1], result_coord[0]), (result_coord[1]+window_size[0],result_coord[0]+window_size[1]), (255,0,0), 2)

    # display final result 
    cv2.imshow("final result", temp2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print ("saving template matching result image")
    cv2.imwrite(os.path.join(pre , "task2_final_result.png"), temp2)
    print ("completed")



#=========
#  MAIN
#=========

print("===========")
print("  TASK 1")
print("===========")
task1()
print("\n===========")
print("  TASK 2")
print("===========")
task2()


