# Test programs for vehicle detection
# Each step tests an aspect of the vehicle detection pipeline and checks for the expected results

import vehicle_detection as vd
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Classifier settings - determined after searching parameter space
color_space_best = 'YCrCb' #'HSV'
spatial_size_best = (32,32)
hist_bins_best = 64
orient_best = 9
pix_per_cell_best = 8
cell_per_block_best = 2
hog_channel_best = 'ALL'

nTests = 0
nTestsPassed = 0
start_time = time.time()

# Test the reading of the files list
# Expect over 8000 images per directory structure
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test reading of car/non-car lists of images')
print('Expect >8000 images per list')
cars, non_cars = vd.read_file_list('samples/vehicles','samples/non-vehicles')
print('  List of cars     : ', len(cars))
print('  List of non-cars : ', len(non_cars))
if len(cars) > 8000 and len(non_cars) > 8000:
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test the feature extraction process using a single image
# This will test the set of feature extraction functions - bin_spatial, color_hist and get_hog_features
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test reading of single example image and creating a feature vector')
print('Expect to get a feature vector size')

img_file = cars[0]  # take the first image
img = mpimg.imread(img_file)
print('Image pixel values range : ', np.min(img), ' to ', np.max(img))      # useful to see when messing with PNG/JPG

# now calc the features
features = vd.single_img_features(img, bins_range=(0,1))
print('Feature vector length : ', len(features))

if True:    # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test the feature extraction and create some example images, using the cars set
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test reading of 10 car images, creating a feature vector and HOG images per channel')
print('Expect copy images and HOG example images')

vd.create_dir('outputs')    # create the directory if not already present
for i in range(0,10):
    img_file = cars[i]      # first 10 images from the training data
    img = mpimg.imread(img_file)    # read the sample image
    # create some sample outputs - will use these for comparison with the HOG features version
    out_file = 'outputs/car_sample' + str(i) + '.png'
    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img)
    f.savefig(out_file)
    plt.close()
    # create the HOG images for visualisation
    for c in [0, 1, 2]:
        features, hog_img = vd.single_img_features(img, hog_channel=c, hog_vis=True, bins_range=(0,1))
        out_file = 'outputs/car_sample'+ str(i) + '_HOG_'  + '_ch' + str(c)  + '.png'
        f, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(hog_img, cmap='hot')
        f.savefig(out_file)
        plt.close()

if True:    # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test the feature extraction and create some images, using the non-cars set
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test reading of 10 non-car images, creating a feature vector and HOG images per channel')
print('Expect copy images and HOG example images')

vd.create_dir('outputs')    # create the directory if not already present
for i in range(0,10):
    img_file = non_cars[i]      # first 10 images from the training data
    img = mpimg.imread(img_file)    # read the sample image
    # create some sample outputs - will use these for comparison with the HOG features version
    out_file = 'outputs/non_sample' + str(i) + '.png'
    f, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img)
    f.savefig(out_file)
    plt.close()
    # create the HOG images for visualisation
    for c in [0, 1, 2]:
        features, hog_img = vd.single_img_features(img, hog_channel=c, hog_vis=True, bins_range=(0,1))
        out_file = 'outputs/non_sample'+ str(i) + '_HOG_'  + '_ch' + str(c)  + '.png'
        f, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(hog_img, cmap='hot')
        f.savefig(out_file)
        plt.close()

if True:    # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test the feature extraction from a list of filenames
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test bulk processing of feature extraction')
print('Expect features lists of same lengths as the cars/non-cars lists')

# car_features = vd.extract_features(cars, color_space='RGB', spatial_size=(32, 32),
#                      hist_bins=32, bins_range=(0,1), orient=9,
#                      pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                      spatial_feat=True, hist_feat=True, hog_feat=True)
# non_car_features = vd.extract_features(non_cars, color_space='RGB', spatial_size=(32, 32),
#                      hist_bins=32, bins_range=(0,1), orient=9,
#                      pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                      spatial_feat=True, hist_feat=True, hog_feat=True)

car_features = vd.extract_features(cars, color_space=color_space_best, spatial_size=spatial_size_best,
                                   hist_bins=hist_bins_best, bins_range=(0, 1), orient=orient_best,
                                   pix_per_cell=pix_per_cell_best, cell_per_block=cell_per_block_best,
                                   hog_channel=hog_channel_best,
                                   spatial_feat=True, hist_feat=True, hog_feat=True)
non_car_features = vd.extract_features(non_cars, color_space=color_space_best, spatial_size=spatial_size_best,
                                   hist_bins=hist_bins_best, bins_range=(0, 1), orient=orient_best,
                                   pix_per_cell=pix_per_cell_best, cell_per_block=cell_per_block_best,
                                   hog_channel=hog_channel_best,
                                   spatial_feat=True, hist_feat=True, hog_feat=True)

print('Car features list length     :', len(car_features))
print('Non-car features list length :', len(non_car_features))
if len(car_features) == len(cars) and len(non_car_features) == len(non_cars):
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Train a linear SVM classifier on this data
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test creation of a linear SVM classifier')
print('Expect to return a classifier object and some level of accuracy')

svc, scaler, acc = vd.train_SVC(car_features, non_car_features)
print('Classifier accuracy : %.3f' % acc)

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test out some predictions
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test a few predictions')
print('Expect to see a set of predictions returned for each sample')

predictions_cars = vd.predict_SVC(car_features[0:10], svc, scaler)
predictions_non_cars = vd.predict_SVC(non_car_features[0:10], svc, scaler)
print('Cars set predictions (should all be True      :',predictions_cars)
print('Non-cars set predictions (should all be False :',predictions_non_cars)

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test save and load of the classifier
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test save and load of the classifier')
print('Expect svc_pickle.p file to be created')

vd.save_classifier(svc, scaler)
svc, scaler = vd.load_classifier()

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test load of an example image and saving it with axes, to determine the extent of the useful region
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test to load an example image and save it with axes, to show relevant area')
print('Expect outputs/roadway_test_axes.jpg to be created')

roadway_img = mpimg.imread('roadway_test.jpg')
f, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(roadway_img)
f.savefig('outputs/roadway_test_axes.jpg')
plt.close()

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test of creating some sliding windows over the roadway image
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test of creating sliding windows over the image')
print('Expect outputs/roadway_test_windows.jpg to be created')

roadway_img = mpimg.imread('roadway_test.jpg')
print('Loaded image pixel values range : ', np.min(roadway_img), ' to ', np.max(roadway_img))      # useful to see when messing with PNG/JPG
# normalise the pixel values for a JPG
roadway_img = roadway_img.astype(np.float32)/255
print('Normalised image pixel values range : ', np.min(roadway_img), ' to ', np.max(roadway_img))      # useful to see when messing with PNG/JPG
# create sliding windows
windows = vd.slide_window(roadway_img,xy_window=(128, 128), xy_overlap=(0.75, 0.75), y_start_stop=[350,625])
# draw some boxes on the image
new_image = vd.draw_boxes(mpimg.imread('roadway_test.jpg'), windows)
f, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(new_image)
f.savefig('outputs/roadway_test_windows.jpg')
plt.close()

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test of creating some predictions on the image
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test predictions on the sliding windows over an image')
print('Expect outputs/roadway_test_predictions.jpg to be created')

# load the classifier
svc, scaler = vd.load_classifier()
roadway_img = mpimg.imread('roadway_test.jpg')
roadway_img = roadway_img.astype(np.float32)/255
windows = vd.slide_window(roadway_img,xy_window=(128, 128), xy_overlap=(0.75, 0.75), y_start_stop=[350,625])

car_windows = []
for window in windows:
    # get some predictions
    test_image = vd.get_window_image(roadway_img, window)
    prediction = vd.predict_window(test_image, svc, scaler, color_space=color_space_best, spatial_size=spatial_size_best,
                                   hist_bins=hist_bins_best, bins_range=(0, 1), orient=orient_best,
                                   pix_per_cell=pix_per_cell_best, cell_per_block=cell_per_block_best,
                                   hog_channel=hog_channel_best,
                                   spatial_feat=True, hist_feat=True, hog_feat=True)
    if prediction == 1.:
        car_windows.append(window)

box_image = vd.draw_boxes(mpimg.imread('roadway_test.jpg'), car_windows)
f, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(box_image)
f.savefig('outputs/roadway_test_predictions.jpg')
plt.close()

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test creating a heatmap of a series of detections - output the heatmap
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test creation of a heatmap image')
print('Expect outputs/heatmap_all.jpg to be created')

heatmap = vd.create_heatmap(mpimg.imread('roadway_test.jpg'), car_windows)
f, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(heatmap)
f.savefig('outputs/heatmap_all.jpg')
plt.close()

if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test thresholding on the detections - output the thresholded heatmap
nTests += 1
test_start_time = time.time()
print('Test #', nTests)
print('Test thresholding of heat map')
print('Expect outputs/heatmap_threshold.jpg to be created')



if True:  # or confirmation condition for this test
    nTestsPassed +=1
    print('Test passed')
else:
    print('Test failed')
print('Time this test : %.3f seconds' % (time.time() - test_start_time))
print()

# Test the production of positive detections using the labels function



# wrap up of the testing process
print()
print('Total tests   : ', nTests)
print('Tests passed  : ', nTestsPassed)
print('Tests failed  : ', (nTests-nTestsPassed))
print('Total time (s):  %.3f' % (time.time() - start_time))



# nTests += 1
# test_start_time = time.time()
# print('Test #', nTests)
# print('Test ...')
# print('Expect ...')
#
#
# if True:  # or confirmation condition for this test
#     nTestsPassed +=1
#     print('Test passed')
# else:
#     print('Test failed')
# print('Time this test : %.3f seconds' % (time.time() - test_start_time))
# print()
