import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# Camshift

def skeleton_tracker1(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return


    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt = (0,c+w/2,r+h/2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)
    
    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        # write the result to the output file
        pt = (frameCounter,x+w/2,y+h/2)
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

		
        frameCounter = frameCounter + 1

    output.close()

def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]



#optical flow tracker

def skeleton_tracker4(v, file_name):
    
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    feature_params = dict( maxCorners = 100,
                                     qualityLevel = 0.1,
                                     minDistance = 7,
                                     blockSize = 7 )

    frameCounter = 0    
    lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret,old_frame = v.read()
    x,y,w,h=detect_one_face(old_frame)
    pt = (0,x+w/2,y+h/2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
	
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    crop_img = old_gray[y:y+h, x:x+w] 
    p0 = cv2.goodFeaturesToTrack(crop_img, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
   

    while(1):
        ret,frame = v.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #crop_img = frame_gray[y:y+h, x:x+w] 

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        pt = (frameCounter,a,b)
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
		
    cv2.destroyAllWindows()
    v.release()

    output.close()







# Particle Filter

def skeleton_tracker2(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt = (0,c+w/2,r+h/2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    n_particles = 200
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    stepsize = 15
 
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        
        hsvt = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)	
        hist_bp = cv2.calcBackProject([hsvt],[0],roi_hist,[0,180],1)
        
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int) 
        
        f = particleevaluator(hist_bp, particles.T)
      
        weights = np.float32(f.clip(1))
        weights /= np.sum(weights)
        if 1. / np.sum(weights**2) < n_particles / 2.:
             particles = particles[resample(weights),:] 
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average        
        pt = (frameCounter,pos[0],pos[1])
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        #cv2.imshow('img2',frame)
 
	
        for i in particles:
 
        	img2 = cv2.circle(frame,(i[0],i[1]),3,255,1)
        	cv2.imshow('img2',img2)
      
        k = cv2.waitKey(25) & 0xff
        if k == 27:
           	break
        else:
            	cv2.imwrite(chr(k)+".jpg",img2) 		
        frameCounter = frameCounter + 1
      
    output.close()



#kalman filter
def skeleton_tracker3(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return


    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt = (0,c+w/2,r+h/2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman = cv2.KalmanFilter(4,2,0)	
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    measurement = np.array([c+w/2, r+h/2], dtype='float64')
   
    
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        
        prediction = kalman.predict() #prediction
        x,y,w,h = detect_one_face(frame) #checking measurement
        measurement = np.array([x+w/2, y+h/2], dtype='float64')
            
        if not (x ==0 and y==0 and w==0 and h ==0):
            posterior = kalman.correct(measurement)
        if x ==0 and y==0 and w==0 and h ==0:
            x,y,w,h = prediction
        else:
            x,y,w,h = posterior	
        pt = (frameCounter,x+w/2,y+h/2)
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        img2 = cv2.rectangle(frame, (int(x),int(y)), (int(x+3),int(y+3)), 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

		
        frameCounter = frameCounter + 1

    output.close()




if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);
    if (question_number == 1):
        skeleton_tracker1(video, "output_camshift.txt")
    elif (question_number == 2):
        skeleton_tracker2(video, "output_particle.txt")
    elif (question_number == 3):
        skeleton_tracker3(video, "output_kalman.txt")
    elif (question_number == 4):
        skeleton_tracker4(video, "output_of.txt")

'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''

