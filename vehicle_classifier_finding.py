# Find the best options across the various sets of parameters, by training with different options combination
# Neil Maude, Feb 2017

# There are potentially many options and each takes a minute or so to train
# Hence just taking a look at the colour spaces and whether extra spatial size or bins helps
# Also comparing a single HOG channel vs ALL

import vehicle_detection as vd
import time

# Get the lists
cars, non_cars = vd.read_file_list('samples/vehicles','samples/non-vehicles')

color_space_options = ['YCrCb','RGB','HSV','HLS']    # fails within the fit process for LUV and YUV...
spatial_size_options = [(32,32), (16,16) ] #[(32,32), (64,64)]
hist_bins_options = [64, 32]
orient_options = [9]      # from lecture by Navneet Dalal, should be best around 9-ish
pix_per_cell_options = [8]
cell_per_block_options = [2]
hog_channel_options = ['ALL',0] # [0,'ALL']

num_options = len(color_space_options) * len(spatial_size_options) * len(hist_bins_options) * len(orient_options)
num_options *= len(pix_per_cell_options) * len(cell_per_block_options) * len(hog_channel_options)

print('Starting options test')
print('Total number of options : ', num_options)
print()

best_acc = 0.0
best_settings = []
start_time = time.time()

for c in color_space_options:
    for s in spatial_size_options:
        for h in hist_bins_options:
            for o in orient_options:
                for p in pix_per_cell_options:
                    for cb in cell_per_block_options:
                        for hc in hog_channel_options:
                            print('Training classifier on  : ', c, ' ', s, ' ', h, ' ', o, ' ', p, ' ', cb, ' ', hc)
                            test_start_time = time.time()
                            # read the features
                            car_features = vd.extract_features(cars, color_space=c, spatial_size=s,
                                                 hist_bins=h, bins_range=(0,1), orient=o,
                                                 pix_per_cell=p, cell_per_block=cb, hog_channel=hc,
                                                 spatial_feat=True, hist_feat=True, hog_feat=True)
                            non_car_features = vd.extract_features(non_cars, color_space=c, spatial_size=s,
                                                 hist_bins=h, bins_range=(0,1), orient=o,
                                                 pix_per_cell=p, cell_per_block=cb, hog_channel=hc,
                                                 spatial_feat=True, hist_feat=True, hog_feat=True)
                            # train a classifier
                            svc, scaler, acc = vd.train_SVC(car_features, non_car_features)
                            print('Accuracy : %.3f' % acc)
                            print('Time this test : %.3f seconds' % (time.time() - test_start_time))
                            if acc > best_acc:
                                print('Best so far...')
                                best_acc = acc
                                best_settings = [c,s,h,o,p,cb,hc]
print('Completed run')
print()
print('Total time (s):  %.3f' % (time.time() - start_time))
print('Best accuracy : %.3f' % best_acc)
print('Best settings : ', best_settings)
