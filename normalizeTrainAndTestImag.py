from PIL import Image
import os

NORMALIZED_WID = 224
NORMALIZED_HEI = 224
BASE_PATH = 'my_directory/'
TARGET_PATH = 'normalized_directory'


def saveFile(image, targetPath, fileName):
    newFileName = 'normalized_'+ fileName
    image.save(f"{targetPath}/{newFileName}")

def getImage(basePath, fileName):
    image = Image.open(basePath + fileName)
    print(f"in: {image.size}")
    image.show()
    return image

def resize_scale_image_by_box(image, box):
    outImage = image.crop(box)
    outImage.thumbnail((NORMALIZED_WID, NORMALIZED_HEI))
    print(f"out: {outImage.size}")
    outImage.show()
    return outImage

def calculate_image_crop_box_by_center(image):
    # box=(left, upper, right, lower)
    width, height = image.size # Output: (499, 375)
    center_x = width/2
    center_y = height/2
    if (width <= height):
        top = center_y - width/2
        left = 0
        return (left, top, left + width, top + width)
    else:
        top = 0
        left = center_x - height/2
        return (left, top, left + height, top + height)
        
def calculate_image_crop_box_by_left_top(image):
    width, height = image.size # Output: (499, 375)
    if (width <= height):
        return (0, 0, width, width)
    else:
        return (0, 0, height, height)

def calculate_image_crop_box_by_random(image, boxCount):
    boxes = []
    width, height = image.size # Output: (499, 375)
    if (width <= height):
        difference = (height - width)/boxCount
        for x in range(boxCount):
            left = 0
            top = difference*x
            box = (left, top, left + width, top + width)
            boxes.append(box)
    else:
        difference = (width - height)/boxCount
        print(difference)
        for x in range(boxCount):
            left = difference*x
            top = 0
            box = (left, top, left + height, top + height)
            print(box)
            boxes.append(box)
    return boxes

def normalize_training_images() :
    with os.scandir(BASE_PATH) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = entry.name
                image = getImage(BASE_PATH, filename)
                croppedBox = calculate_image_crop_box_by_center(image)
                #croppedBox = calculate_image_crop_box_by_left_top(image)
                image = resize_scale_image_by_box(image, croppedBox)
                saveFile(image, TARGET_PATH, filename)

def normalize_single_image(image):
    cropped_images = []
    cropped_boxes = calculate_image_crop_box_by_random(image, 5)
    for box in cropped_boxes:
        cropped_images.append(resize_scale_image_by_box(image, box))
    return cropped_images

normalize_training_images()
#normalize_single_image(Image.open('cat.1436.jpg'))