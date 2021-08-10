import glob
import os


def find_img_differences(file_dir_1, file_dir_2, delete=False):
    """Finds similar image names and prints them out. Deletes from one hand and the other to even out the images"""
    #Implement a feature to delete the img without an xml file if possible
    test_files = glob.glob(file_dir_1)
    train_files = glob.glob(file_dir_2)

    test_set = set()
    for item in test_files:
        item = os.path.basename(item)
        test_set.add(item)

    train_set = set()
    for item in train_files:
        item = os.path.basename(item)
        train_set.add(item)

    intersect = train_set.intersection(test_set)
    if len(intersect) == 0:
        print("The file names are all unique.")
    else:
        if delete:
            rotate = 0
            for item in intersect:
                if rotate == 0:
                    rotate = 1
                    trash_test_file = os.path.abspath(os.path.join(test_dir, '..'))
                    trash_test_file = os.path.abspath(os.path.join(trash_test_file, item))
                    os.remove(trash_test_file)
                else:
                    rotate = 0
                    trash_train_file = os.path.abspath(os.path.join(train_dir, '..'))
                    trash_train_file = os.path.abspath(os.path.join(trash_train_file, item))
                    os.remove(trash_train_file)
                print("{} has been deleted.".format(item))
        else:
            for item in intersect:
                print(item)

        print("There are some intersecting image names.")


test_dir = '/Users/ccampos/Desktop/models/research/object_detection/pv-learning/images/test/*.png'
#test_dir = '/Users/ccampos/Desktop/image_results'
train_dir = '/Users/ccampos/Desktop/models/research/object_detection/pv-learning/images/train/*.png'


find_img_differences(test_dir, train_dir, delete=True)


