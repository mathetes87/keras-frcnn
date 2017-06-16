# relative import imgaug library files
# first add folder to python path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from imgaug.imgaug import imgaug as ia
from imgaug.imgaug import augmenters as iaa

# MODIFY BELOW PARAMETERS AS NEEDED
# show augmented images and bboxes
# useful for testing different parameters below
def get_show_parameters():
    SHOW_BBOXES = True
    SHOW_AUGMENTED_IMAGE_SAMPLES = True
    return SHOW_BBOXES, SHOW_AUGMENTED_IMAGE_SAMPLES

# tune this parameters to get different augmentation results
# more details here: https://github.com/aleju/imgaug
# credits mostly to https://github.com/aleju for the amazing library
def get_augment_parameters():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    parameters = [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip a percentage of all images
            iaa.Flipud(0.2), # vertically flip a percentage of all images
            sometimes(iaa.Crop(percent=(0, 0.05))), # crop images by percentage of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to percentage of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by percent (per axis)
                rotate=(-15, 15), # rotate by degrees
                shear=(-10, 10), # shear by degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples in imgaug repo)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 0.5), n_segments=(100, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma value
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0)), # emboss images
                    # search either for all edges or for directed edges
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.5)),
                        iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove percentage of the pixels
                        iaa.CoarseDropout((0.03, 0.1), size_percent=(0.02, 0.1), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.1, 2.0), sigma=0.2)), # move pixels locally around
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))) # sometimes move parts of the image around
                ],
                random_order=True
            )
        ]
    return parameters
