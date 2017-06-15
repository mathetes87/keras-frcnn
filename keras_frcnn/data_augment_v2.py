import cv2
import numpy as np
import copy
import os,sys,inspect
from scipy import misc
# relative import imgaug library files
# first add folder to python path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from imgaug.imgaug import imgaug as ia
from imgaug.imgaug import augmenters as iaa
import data_augment_v2_parameters as augment_parameters

# parameters for test purposes
SHOW_KEYPOINTS, SHOW_AUGMENTED_IMAGE_SAMPLES = augment_parameters.get_show_parameters()

def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    # add dimension to keep imgaug tutorial code unchanged
    images = np.expand_dims(cv2.imread(img_data_aug['filepath']), axis=0)
    
    if augment:
        # The augmenters expect a list of imgaug.KeypointsOnImage.
        keypoints_on_images = []
        for image in images:
            keypoints = []
            for bbox in img_data_aug['bboxes']:
                # for every bbox x1, y1, x2, y2 in pairs
                x1 = bbox['x1']
                y1 = bbox['y1']
                keypoints.append(ia.Keypoint(x=x1, y=y1))
                
                x2 = bbox['x2']
                y2 = bbox['y2']
                keypoints.append(ia.Keypoint(x=x2, y=y2))
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))
        
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        # CHANGE PARAMETERS IN DATA_AUGMENT_V2_PARAMETERS FILE
        seq = iaa.Sequential(
            augment_parameters.get_augment_parameters(),
            random_order=True
        )
        
        # maybe show image transformation sample in grid
        if SHOW_AUGMENTED_IMAGE_SAMPLES:
            seq.show_grid(images, cols=8, rows=8)
        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        
        # augment keypoints and images
        images_aug = seq_det.augment_images(images)
        keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)
        
        # maybe show keypoints
        if SHOW_KEYPOINTS:        
            for img_idx, (image_before, image_after, keypoints_before, keypoints_after) in enumerate(zip(images, images_aug, keypoints_on_images, keypoints_aug)):
                image_before = keypoints_before.draw_on_image(image_before)
                image_after = keypoints_after.draw_on_image(image_after)
                misc.imshow(np.concatenate((image_before, image_after), axis=1)) # before and after
                for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
                    keypoint_old = keypoints_on_images[img_idx].keypoints[kp_idx]
                    x_old, y_old = keypoint_old.x, keypoint_old.y
                    x_new, y_new = keypoint.x, keypoint.y
                    print("[Keypoints for image #%d] before aug: x=%d y=%d | after aug: x=%d y=%d" % (img_idx, x_old, y_old, x_new, y_new))
        
        # update bboxes coordinates after augmentation
        keypoints = keypoints_aug[0].get_coords_array()
        for i, bbox in enumerate(img_data_aug['bboxes']):
            bbox['x1'] = keypoints[2*i][0]
            bbox['y1'] = keypoints[2*i][1]
            bbox['x2'] = keypoints[2*i+1][0]
            bbox['y2'] = keypoints[2*i+1][1]
    
    img = images_aug[0]    
    
    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img
    
if __name__ == "__main__":
    augment()
