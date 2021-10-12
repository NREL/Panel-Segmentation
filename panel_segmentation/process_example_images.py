import os
from detecto import core, utils
import torch
from detecto.utils import read_image, reverse_normalize, \
                normalize_transform, _is_iterable
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

utils.xml_to_csv('C:/Users/kperry/Documents/source/repos/Panel-Segmentation/panel_segmentation/examples/sample_data_set/annotations/',
                  'C:/Users/kperry/Documents/source/repos/Panel-Segmentation/panel_segmentation/examples/sample_data_set/annotations.csv')


import glob

annotations_df = pd.read_csv("C:/Users/kperry/Documents/source/repos/Panel-Segmentation/panel_segmentation/examples/sample_data_set/annotations.csv")
imgs = glob.glob("C:/Users/kperry/Documents/source/repos/Panel-Segmentation/panel_segmentation/examples/sample_data_set/images/*.png")

for img in imgs:
    sys = img.split("\\")[-1]
    file_name_save = img.replace("/images\\", "/labeled_images/")
    #Get the label annotations
    annotations_sub_df = annotations_df[annotations_df['filename'] == sys]
    boxes = list()
    for index, row in annotations_sub_df.iterrows():
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        box   = [xmin,ymin,xmax,ymax]
        boxes.append(box)
    boxes = torch.tensor(boxes)
    labels = list(annotations_sub_df['class'])
    image = read_image(img)
    #Generate the image associated with the classifications
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
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)
    if file_name_save != None:
        plt.savefig(file_name_save)
    plt.show()


    