"""
This script contains the PanelDetection class, which contains
the routines for generating satellite images and running DL and
CV routines on images to get array azimuth and mounting type/configuration.
"""

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm
from PIL import Image
from os import path
import requests
from detecto import core
from detecto.utils import read_image, reverse_normalize, \
    _is_iterable
import matplotlib.patches as patches
import torch
from tensorflow.keras.preprocessing import image as imagex
from torchvision import transforms
from torchvision.ops import nms

panel_seg_model_path = path.join(path.dirname(__file__),
                                 'models',
                                 'VGG16Net_ConvTranpose_complete.h5')
panel_classification_model_path = path.join(path.dirname(__file__),
                                            'models',
                                            'VGG16_classification_model.h5')
mounting_classification_path = path.join(path.dirname(__file__),
                                         'models',
                                         'object_detection_model.pth')


class PanelDetection:
    '''
    A class for training a deep learning architecture,
    detecting solar arrays from a satellite image, performing spectral
    clustering, predicting azimuth, and classifying mounting type and
    configuration.
    '''

    def __init__(self, model_file_path=panel_seg_model_path,
                 classifier_file_path=panel_classification_model_path,
                 mounting_classifier_file_path=mounting_classification_path):
        # This is the model used for detecting if there is a panel or not
        self.classifier = load_model(classifier_file_path,
                                     custom_objects=None,
                                     compile=False)
        self.model = load_model(model_file_path,
                                custom_objects=None,
                                compile=False)
        self.mounting_classifier = core.Model.load(
            mounting_classifier_file_path, ["ground-fixed",
                                            "carport-fixed",
                                            "rooftop-fixed",
                                            "ground-single_axis_tracker"])

    def generateSatelliteImage(self, latitude, longitude,
                               file_name_save, google_maps_api_key):
        """
        Generates satellite image via Google Maps, using a set of lat-long
        coordinates.

        Parameters
        -----------
        latitude: float
            Latitude coordinate of the site.
        longitude: float
            Longitude coordinate of the site.
        file_name_save: string
            File path that we want to save
            the image to, where the image is saved as a PNG file.
        google_maps_api_key: string
            Google Maps API Key for
            automatically pulling satellite images.

        Returns
        -----------
            Figure
            Figure of the satellite image
        """
        # Check input variable for types
        if type(latitude) != float:
            raise TypeError("latitude variable must be of type float.")
        if type(longitude) != float:
            raise TypeError("longitude variable must be of type float.")
        if type(file_name_save) != str:
            raise TypeError("file_name_save variable must be of type string.")
        if type(google_maps_api_key) != str:
            raise TypeError("google_maps_api_key variable must be "
                            "of type string.")
        # Build up the lat_long string from the latitude-longitude coordinates
        lat_long = str(latitude) + ", " + str(longitude)
        # get method of requests module
        # return response object
        r = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap?maptype"
            "=satellite&center=" + lat_long +
            "&zoom=18&size=35000x35000&key="+google_maps_api_key,
            verify=False)
        # Raise an exception if image is not successfully returned
        if r.status_code != 200:
            raise ValueError("Response status code " +
                             str(r.status_code) +
                             ": Image not pulled successfully from API.")
        # wb mode is stand for write binary mode
        with open(file_name_save, 'wb') as f:
            f.write(r.content)
            # close method of file object
            # save and close the file
            f.close()
        # Read in the image and return it via the console
        return Image.open(file_name_save)

    def classifyMountingConfiguration(self, image_file_path,
                                      acc_cutoff=.65,
                                      file_name_save=None,
                                      use_nms=True):
        """
        This function is used to detect and classify the mounting configuration
        of solar installations in satellite imagery. It leverages the Detecto
        package's functionality
        (https://detecto.readthedocs.io/en/latest/api/index.html),
        to perform object detection on a satellite image.

        Parameters
        -----------
        image_file_path: string
            File path of the image. PNG file.
        acc_cutoff: float
            Default set to 0.65. Confidence cutoff for whether or not to
            count a object detection classification as real. All returned
            classications greater than or equal to the accuracy cutoff are
            counted, and all classifications less than the accuracy cutoff
            are thrown out.
        Returns
        -----------
        Returns
            tuple
            Tuple consisting of (scores, labels, boxes), where 'scores' is
            the list of object detection confidence scores, 'labels' is a
            list of all corresponding labels, and 'boxes' is a tensor object
            containing the associated boxes for the associated labels and
            confidence scores.
        """
        image = read_image(image_file_path)
        labels, boxes, scores = self.mounting_classifier.predict(image)
        if use_nms:
            # Perform non-maximum suppression (NMS) on the detections
            nms_idx = nms(boxes=boxes,
                          scores=scores,
                          iou_threshold=0.25)
            scores = scores[nms_idx]
            boxes = boxes[nms_idx]
            labels = [labels[nms] for nms in nms_idx]
        mask = [float(x) > acc_cutoff for x in scores]
        scores = list(np.array(scores)[mask])
        labels = list(np.array(labels)[mask])
        boxes = torch.tensor(np.array(boxes)[mask])
        # This code is adapted from the Detecto package's show_labeled_image()
        # function. See the following link as reference:
        # https://github.com/alankbi/detecto/blob/master/detecto/visualize.py
        # Generate the image associated with the classifications
        fig, ax = plt.subplots(1)
        # If the image is already a tensor, convert it back to a PILImage
        # and reverse normalize it
        if isinstance(image, torch.Tensor):
            image = reverse_normalize(image)
            image = transforms.ToPILImage()(image)
        ax.imshow(image)
        # Show a single box or multiple if provided
        if boxes.ndim == 1:
            boxes = boxes.view(1, 4)
        if labels is not None and not _is_iterable(labels):
            labels = [labels]
        # Plot each box
        for i in range(boxes.shape[0]):
            box = boxes[i]
            width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
            initial_pos = (box[0].item(), box[1].item())
            rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                     edgecolor='r', facecolor='none')
            if labels:
                ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]),
                        color='red')
            ax.add_patch(rect)
        if file_name_save is not None:
            plt.savefig(file_name_save)
        return (scores, labels, boxes)

    def diceCoeff(self, y_true, y_pred, smooth=1):
        """
        This function is used as the metric of similarity between the
        predicted mask and ground truth.

        Parameters
        -----------
        y_true: numpy array of floats
            The true mask of the image
        y_pred: numpy array  of floats
            the predicted mask of the data
        smooth: int
            A parameter to ensure we are not dividing by zero and also
            a smoothing parameter for back -ropagation. If the prediction
            is hard threshold to 0 and 1, it is difficult to back-propagate
            the dice loss gradient. We add this parameter to actually
            smooth out the loss function, making it differentiable.

        Returns
        -----------
        dice: float
            Retuns the metric of similarity between prediction
            and ground truth
        """
        # Ensure that the inputs are of the correct type
        if type(y_true) != np.ndarray:
            raise TypeError("Variable y_true should be of type np.ndarray.")
        if type(y_pred) != np.ndarray:
            raise TypeError("Variable y_pred should be of type np.ndarray.")
        if type(smooth) != int:
            raise TypeError("Variable smooth should be of type int.")
        # If variable types are correct, continue with function
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice

    def diceCoeffLoss(self, y_true, y_pred):
        """
        This function is a loss function that can be used when training
        the segmentation model. This loss function can be used in place
        of binary crossentropy, which is the current loss function in
        the training stage.

        Parameters
        -----------
        y_true: numpy array of floats
            The true mask of the image
        y_pred: numpy array of floats
            The predicted mask of the data

        Returns
        -----------
            float
            The loss metric between prediction and ground truth
        """
        # Ensure that the inputs are of the correct type
        if type(y_true) != np.ndarray:
            raise TypeError("Variable y_true should be of type np.ndarray.")
        if type(y_pred) != np.ndarray:
            raise TypeError("Variable y_pred should be of type np.ndarray.")
        return 1-self.dice_coef(y_true, y_pred)

    def testBatch(self, test_data,
                  test_mask=None,
                  batch_size=16,
                  model=None):
        """
        This function is used to predict the mask of a batch of test
        satellite images. Use this to test a batch of images greater than 4.

        Parameters
        -----------
        test_data: nparray float
            The satellite images
        test_mask: nparray int or float
            The mask ground truth corresponding to the test_data
        batch_size: int
            The batch size of the test_data.
        model: tf.keras.model.object
            A custom model can be provided as input or we can
            use the initialized model

        Returns
        -----------
        test_res: nparray float
            The predicted masks
        accuracy: float
            The accuarcy of prediction as compared with the ground
            truth if provided
        """
        # Ensure that the inputs are of the correct type
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data should be of type np.ndarray.")
        if type(batch_size) != int:
            raise TypeError("Variable batch_size should be of type int.")
        test_datagen = image.ImageDataGenerator(rescale=1./255,
                                                dtype='float32')
        test_image_generator = test_datagen.flow(
            test_data,
            batch_size=batch_size, shuffle=False)
        if model is not None:
            test_res = model.predict(test_image_generator)
        else:
            test_res = self.model.predict(test_image_generator)
        if test_mask is not None:
            test_mask = test_mask/np.max(test_mask)
            accuracy = self.dice_coef(test_mask, test_res)
            return test_res, accuracy
        else:
            return test_res

    def testSingle(self, test_data,
                   test_mask=None,
                   model=None):
        """
        This function is used to predict the mask corresponding
        to a single test image. It takes as input the test_data
        (a required parameter) and two non-required parameters-
        test_mask and model. Use this to test a single image.

        Parameters
        -----------
        test_data: nparray int or float
            The satellite image. dimension is (640,640,3) or
            (a,640,640,3)
        test_mask: nparray int or float
            The ground truth of what the mask should be.
        model: tf.keras model object
            A custom model can be provided as input or
            we can use the initialized model

        Returns
        -----------
        test_res: nparray float
            The predicted mask of the single image.
            The dimension is (640,640 or (a,640,640))
        accuracy: float
            The accuracy of prediction as compared
            with the ground truth if provided
        """
        # Check that the inputs are correct
        if type(test_data) != np.ndarray:
            raise TypeError(
                "Variable test_data must be of type numpy ndarray.")
        # Test that the input array has 2 to 3 channels
        if (len(test_data.shape) > 3) | (len(test_data.shape) < 2):
            raise ValueError(
                "numpy array test_data shape should be 2 or 3 dimensions.")
        # Once the array passes checks, run the sequence
        test_data = test_data/255
        test_data = test_data[np.newaxis, :]
        if model is not None:
            test_res = model.predict(test_data)
        else:
            test_res = self.model.predict(test_data)
            test_res = (test_res[0].reshape(640, 640))
        if test_mask is not None:
            test_mask = test_mask/np.max(test_mask)
            accuracy = self.dice_coef(test_mask, test_res)
            return test_res, accuracy
        else:
            return test_res

    def hasPanels(self, test_data):
        """
        This function is used to predict if there is a panel in an image
        or not. Note that it uses a saved classifier model we have trained
        and not the segmentation model.

        Parameters
        -----------
        test_data: nparray float or int
            The satellite image. The shape should be [a,640,640,3] where
            'a' is the number of data or (640,640,3) if it is a single image

        Returns
        -----------
            boolean
            True if solar array is detected in an image,
            and False otherwise.
        """
        # Check that the input is correct
        if type(test_data) != np.ndarray:
            raise TypeError(
                "Variable test_data must be of type numpy ndarray.")
        # Test that the input array has 3 to 4 channels
        if (len(test_data.shape) > 4) | (len(test_data.shape) < 3):
            raise ValueError(
                "numpy array test_data shape should be 3 dimensions "
                "if a single image, or 4 dimensions if a batch of images.")
        test_data = test_data/255
        # This ensures the first dimension is the number of test data to be
        # predicted
        if test_data.ndim == 3:
            test_data = test_data[np.newaxis, :]
        prediction = self.classifier.predict(test_data)
        # index 0 is for no panels while index 1 is for panels
        if prediction[0][1] > prediction[0][0]:
            return True
        else:
            return False

    def detectAzimuth(self, in_img, number_lines=5):
        """
        This function uses canny edge detection to first extract the edges
        of the input image. To use this function, you have to first predict
        the mask of the test image using testSingle function. Then use the
        cropPanels function to extract the solar panels from the input image
        using the predicted mask. Hence the input image to this
        function is the cropped image of solar panels.

        After edge detection, Hough transform is used to detect the most
        dominant lines in the input image and subsequently use that to predict
        the azimuth of a single image.

        Parameters
        -----------
        in_img: nparray uint8
            The image containing the extracted solar panels with other pixels
            zeroed off. Dimension is [1,640,640,3]
        number_lines: int
            This variable tells the function the number of dominant lines it
            should examine. We currently inspect the top 5 lines.

        Returns
        -----------
        azimuth: int
            The azimuth of the panel in the image.
        """
        # Check that the input variables are of the correct type
        if type(in_img) != np.ndarray:
            raise TypeError("Variable in_img must be of type numpy ndarray.")
        if type(number_lines) != int:
            raise TypeError("Variable number_lines must be of type int.")
        # Run through the function
        edges = cv2.Canny(in_img[0], 50, 150, apertureSize=3)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(edges, theta=tested_angles)
        origin = np.array((0, edges.shape[1]))
        ind = 0
        azimuth = 0
        az = np.zeros((number_lines))
        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        for _, angle, dist in \
            zip(*hough_line_peaks(h, theta, d, num_peaks=number_lines,
                                  threshold=0.25*np.max(h))):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)

            deg_ang = int(np.rad2deg(angle))
            if deg_ang >= 0:
                az[ind] = 90+deg_ang
            else:
                az[ind] = 270 + deg_ang
            ind = ind+1
        unique_elements, counts_elements = np.unique(az, return_counts=True)
        check = counts_elements[np.argmax(counts_elements)]
        if check == 1:
            for _, angle, dist in zip(
                    *hough_line_peaks(h, theta, d, num_peaks=1,
                                      threshold=0.25*np.max(h))):
                deg_ang = int(np.rad2deg(angle))
                if deg_ang >= 0:
                    azimuth = 90+deg_ang
                else:
                    azimuth = 270 + deg_ang
        else:
            azimuth = (unique_elements[np.argmax(counts_elements)])
        return azimuth

    def cropPanels(self, test_data, test_res):
        """
        This function basically isolates regions with solar panels in a
        satellite image using the predicted mask. It zeros out other
        pixels that does not contain a panel.
        You can use this for a single test data or multiple test data.

        Parameters
        ----------
        test_data:  nparray float
            This is the input test data. This can be a single image
            or multiple image. Hence the dimension can be (640,640,3)
            or (a,640,640,3)
        test_res:   nparray float
            This is the predicted mask of the test images passed
            as an input and used to crop out the solar panels.
            Dimension is (640,640)

        Returns
        ----------
        new_test_res: nparray uint8
            This returns images here the solar panels have been cropped
            out and the background zeroed. It has the same shape as test
            data.  The dimension is [a,640,640,3] where a is the number of
            input images.
        """
        # Check that the input variables are of the correct type
        if type(test_data) != np.ndarray:
            raise TypeError(
                "Variable test_data must be of type numpy ndarray.")
        if type(test_res) != np.ndarray:
            raise TypeError("Variable test_res must be of type numpy ndarray.")
        # Convert the test_data array from 3D to 4D
        if test_data.ndim == 3:
            test_data = test_data[np.newaxis, :]
        if test_res.ndim == 2:
            test_res = test_res[np.newaxis, :]
        new_test_res = np.uint8(np.zeros((test_data.shape[0], 640, 640, 3)))
        for ju in np.arange(test_data.shape[0]):
            in_img = test_res[ju].reshape(640, 640)
            in_img[in_img < 0.9] = 0
            in_img[in_img >= 0.9] = 1
            in_img = np.uint8(in_img)
            test2 = np.copy(test_data[ju])
            test2[(1-in_img).astype(bool), 0] = 0
            test2[(1-in_img).astype(bool), 1] = 0
            test2[(1-in_img).astype(bool), 2] = 0
            new_test_res[ju] = test2
        return new_test_res

    def plotEdgeAz(self, test_results, no_lines=5,
                   no_figs=1, save_img_file_path=None,
                   plot_show=False):
        """
        This function is used to generate plots of the image with its azimuth
        It can generate three figures or one. For three figures, that include
        the input image, the hough transform space and the input images with
        detected lines. For single image, it only outputs the input image
        with detected lines.

        Parameters
        ----------
        test_results: nparray float64 or unit8
            8-bit input image. This variable represents the predicted images
            from the segmentation model. Hence the dimension must be [a,b,c,d]
            where [a] is the number of images, [b,c] are the dimensions
            of the image - 640 x 640 in this case and [d] is 3 - RGB
        no_lines: int
            default is 10. This variable tells the function the number of
            dominant lines it should examine.
        no_figs: int
            1 or 3. If the number of figs is 1, it outputs the mask with
            Hough lines and the predicted azimuth.
            However, if the number of lines is 3, it gives three plots.
                1. The input image,
                2. Hough transform search space
                3. Unput image with houghlines and the predicted azimuth
        save_img_file_path: string
            You can pass as input the location to save the plots
        plot_show: boolean
            If False, it will supress the plot as an output
            and just save the  plots in a folder

        Returns
        ----------
            Figure
            Plot of the masked image, with detected Hough Lines and azimuth
            estimate.
        """
        # Check that the input variables are of the correct type
        if type(test_results) != np.ndarray:
            raise TypeError(
                "Variable test_results must be of type numpy ndarray.")
        if type(no_lines) != int:
            raise TypeError("Variable no_lines must be of type int.")
        if type(no_figs) != int:
            raise TypeError("Variable no_figs must be of type int.")
        if type(plot_show) != bool:
            raise TypeError("Variable no_figs must be of type boolean.")

        for ii in np.arange(test_results.shape[0]):
            # This changes the float64 to uint8
            if (test_results.dtype is np.dtype(np.float64)):
                in_img = test_results[ii].reshape(640, 640)
                in_img[in_img < 0.9] = 0
                in_img[in_img >= 0.9] = 1
                in_img = np.uint8(in_img)

            in_img = test_results[ii]
            # Edge detection
            edges = cv2.Canny(in_img, 50, 150, apertureSize=3)
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            h, theta, d = hough_line(edges, theta=tested_angles)
            az = np.zeros((no_lines))
            origin = np.array((0, edges.shape[1]))
            ind = 0
            # Generating figure 1
            fig, ax = plt.subplots(1, no_figs, figsize=(10, 6))
            if no_figs == 1:
                ax.imshow(edges)  # cmap=cm.gray)
                for _, angle, dist in zip(
                        *hough_line_peaks(h, theta, d, num_peaks=no_lines,
                                          threshold=0.25*np.max(h))):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                    deg_ang = int(np.rad2deg(angle))
                    if deg_ang >= 0:
                        az[ind] = 90+deg_ang
                    else:
                        az[ind] = 270 + deg_ang
                    ind = ind+1
                    ax.plot(origin, (y0, y1), '-r')
                ax.set_xlim(origin)
                ax.set_ylim((edges.shape[0], 0))
                ax.set_axis_off()
                unique_elements, counts_elements = np.unique(
                    az, return_counts=True)
                check = counts_elements[np.argmax(counts_elements)]
                if check == 1:
                    for _, angle, dist in zip(
                            *hough_line_peaks(h, theta, d, num_peaks=1,
                                              threshold=0.25*np.max(h))):
                        deg_ang = int(np.rad2deg(angle))
                        if deg_ang >= 0:
                            azimuth = 90+deg_ang
                        else:
                            azimuth = 270 + deg_ang
                else:
                    azimuth = (unique_elements[np.argmax(counts_elements)])
                    ax.set_title('Azimuth = %i' % azimuth)
                # Save the image
                if save_img_file_path is not None:
                    plt.savefig(save_img_file_path + 'crop_mask_az_'+str(ii),
                                dpi=300)
                # Show the plot if plot_show = True
                if plot_show is True:
                    plt.tight_layout()
                    plt.show()
            elif no_figs == 3:
                ax = ax.ravel()

                ax[0].imshow(in_img, cmap=cm.gray)
                ax[0].set_title('Input image')
                ax[0].set_axis_off()

                ax[1].imshow(np.log(1 + h),
                             extent=[np.rad2deg(
                                 theta[-1]), np.rad2deg(theta[0]),
                                 d[-1], d[0]],
                             cmap=cm.gray, aspect=1/1.5)
                ax[1].set_title('Hough transform')
                ax[1].set_xlabel('Angles (degrees)')
                ax[1].set_ylabel('Distance (pixels)')
                ax[1].axis('image')

                ax[2].imshow(in_img)  # cmap=cm.gray)
                origin = np.array((0, edges.shape[1]))
                ind = 0
                for _, angle, dist in zip(
                        *hough_line_peaks(h, theta, d,
                                          num_peaks=no_lines,
                                          threshold=0.25*np.max(h))):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)

                    deg_ang = int(np.rad2deg(angle))
                    if deg_ang >= 0:
                        az[ind] = 90+deg_ang
                    else:
                        az[ind] = 270 + deg_ang
                    ind = ind+1
                    ax.plot(origin, (y0, y1), '-r')
                ax[2].set_xlim(origin)
                ax[2].set_ylim((edges.shape[0], 0))
                ax[2].set_axis_off()
                unique_elements, counts_elements = np.unique(
                    az, return_counts=True)

                check = counts_elements[np.argmax(counts_elements)]

                if check == 1:
                    for _, angle, dist in zip(
                            *hough_line_peaks(h, theta, d,
                                              num_peaks=1,
                                              threshold=0.25*np.max(h))):
                        deg_ang = int(np.rad2deg(angle))
                        if deg_ang >= 0:
                            azimuth = 90+deg_ang
                        else:
                            azimuth = 270 + deg_ang
                else:
                    azimuth = (unique_elements[np.argmax(counts_elements)])
                    ax[2].set_title('Azimuth = %i' % azimuth)
                # save the image
                if save_img_file_path is not None:
                    plt.savefig(save_img_file_path + '/crop_mask_az_'+str(ii),
                                dpi=300)
                # Show the plot if plot_show = True
                if plot_show is True:
                    plt.tight_layout()
                    plt.show()
            else:
                print("Enter valid parameters")

    def clusterPanels(self, test_mask, boxes, fig=False):
        '''
        This function uses object detection outputs to cluster the panels

        Parameters
        ----------
        test_mask : Array of booleans
            Size (640, 640, 3). This is the boolean mask output of the
            testSingle() function, where solar array pixels are masked.
        boxes: Pytorch tensor
            Contains the detected boxes found by the object detection model.
            This is the 'boxes' output of the classifyMountingConfiguration()
            function.
        fig : boolean
            shows the clustering image if fig = True

        Returns
        -------
            integer
            Total number of clusters detected in the image
            uint8
            Masked image containing detected clusters each of
            dimension (640,640,3). The masked image is 4D, with the first
            dimension representing the total number of clusters, as follows:
            (number clusters, 640, 640, 3)
        '''
        # Check that the input variables are of the correct type
        if type(test_mask) != np.ndarray:
            raise TypeError(
                "Variable test_mask must be of type numpy ndarray.")
        if type(fig) != bool:
            raise TypeError("Variable fig must be of type bool.")
        # Continue running through the function if all the inputs are correct
        if (len(test_mask.shape) < 3):
            test_mask = cv2.cvtColor(test_mask, cv2.COLOR_GRAY2RGB)
        test_mask = test_mask.reshape(640, 640, 3)
        clusters = list()
        for box in boxes:
            cluster = test_mask[int(box[1]):int(box[3]),
                                int(box[0]):int(box[2]), :]
            result = np.full(test_mask.shape, (0, 0, 0), dtype=np.uint8)
            result[int(box[1]):int(box[3]),
                   int(box[0]):int(box[2]), :] = cluster
            result = result.reshape(((1,) + result.shape))
            clusters.append(result)
        if len(clusters) > 0:
            clusters = np.concatenate(clusters)
        # Plot the individual clusters in the image if fig is True
        if fig is True:
            for idx in range(clusters.shape[0]):
                plt.imshow(clusters[idx], interpolation='nearest')
                plt.title("Cluster #" + str(idx + 1))
                plt.show()
        return clusters.shape[0], clusters

    def runSiteAnalysisPipeline(self,
                                file_name_save_img,
                                latitude=None,
                                longitude=None,
                                google_maps_api_key=None,
                                file_name_save_mount=None,
                                file_path_save_azimuth=None,
                                generate_image=False):
        """
        This function runs a site analysis on a site, when latitude
        and longitude coordinates are given. It includes the following steps:
           1. If generate_image = True, taking a satellite image in
               Google Maps of site location, based on its latitude-longitude
               coordinates. The satellite image is then saved under
               'file_name_save_img' path.
           2. Running the satellite image through the mounting
               configuration/type pipeline. The associated mount predictions
               are returned, and the most frequently occurring mounting
               configuration of the predictions is selected. The associated
               labeled image is stored under the 'file_name_save_mount' path.
           3. Running the satellite image through the azimuth estimation
               algorithm. A default single azimuth is calculated in this
               pipeline for simplicity. The detected azimuth image is saved
               via the file_path_save_azimuth path.
           4. If a mounting configuration is detected as a single-axis
               tracker, an azimuth correction of 90 degrees is applied, as
               azimuth runs parallel to the installation, as opposed to
               perpendicular.
           5. A final dictionary of analysed site metadata is returned,
               including latitude, longitude, detected azimuth, and mounting
               configuration.

        Parameters
        ----------
        file_name_save_img: string
            File path that we want to save the raw satellite image to.
            PNG file.
        latitude: float
            Default None. Latitude coordinate of the site. Not required if
            we're using a pre-generated satellite image.
        longitude: float
            Default None. Longitude coordinate of the site. Not required if
            we're using a pre-generated satellite image.
        google_maps_api_key: string
            Default None. Google Maps API Key for automatically pulling
            satellite images. Not required if we're using a pre-generated
            satellite image.
        file_name_save_mount: string
            File path that we want to save the
            labeled mounting configuration image to. PNG file.
        file_name_save_azimuth: string
            File path that we want to save the
            predicted azimuth image to. PNG file.
        generate_image: bool
            Whether or not we should generate the image via the Google
            Maps API. If set to True, satellite image is generated and
            saved. Otherwise, no image is generated and the image
            saved under the file_name_save_img path is used.

        Returns
        -------
        Python dictionary
            Dictionary containing the latitude, longitude, classified mounting
            configuration, and the estimated azimuth of a site.
        """
        # Generate the associated satellite image, if generate_image
        # is set to True
        if generate_image is True:
            self.generateSatelliteImage(latitude,
                                        longitude,
                                        file_name_save_img,
                                        google_maps_api_key)
        else:
            print("Image not generated. Using image " +
                  str(file_name_save_img) + "...")
        # Run through the mounting configuration pipeline
        (scores, labels, boxes) =\
            self.classifyMountingConfiguration(
                image_file_path=file_name_save_img,
                acc_cutoff=.65,
                file_name_save=file_name_save_mount)
        # Run the azimuth detection pipeline
        x = imagex.load_img(file_name_save_img,
                            color_mode='rgb',
                            target_size=(640, 640))
        x = np.array(x)
        # Mask the satellite image
        res = self.testSingle(x, test_mask=None,  model=None)
        # Use the mask to isolate the panels
        new_res = self.cropPanels(x, res)
        # Cluster components based on the object detection boxes
        no_clusters, clusters = self.clusterPanels(new_res,
                                                   boxes,
                                                   fig=True)
        # Generate a list of all of the azimuths
        az_list = list()
        for idx in range(no_clusters):
            az = self.detectAzimuth(np.expand_dims(clusters[idx], axis=0),
                                    number_lines=5)
            az_list.append(az)
        # Plot edges + azimuth for each cluster
        if len(clusters) > 0:
            self.plotEdgeAz(clusters, 5, 1,
                            save_img_file_path=file_path_save_azimuth)
        # Update azimuth value if the site is a single-axis tracker
        # system. This logic handles single axis tracker cases that are
        # labeled at 90 or 270 (perpendicular to row direction)
        az_list_updated = list()
        mounting_config_list_updated = list()
        for array_number in range(len(az_list)):
            mount_classification = labels[array_number]
            az = az_list[array_number]
            if mount_classification == 'ground-single_axis_tracker':
                if az <= 120:
                    az = az + 90
                elif az > 200:
                    az = az - 90
                else:
                    pass
            az_list_updated.append(az)
            mounting_config_list_updated.append(mount_classification)
        site_analysis_dict = {"associated_azimuths": az_list_updated,
                              "mounting_type": mounting_config_list_updated
                              }
        return site_analysis_dict
