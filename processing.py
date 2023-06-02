import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern


# Showing Data
def show_dataset(image_class, label_name):
    plt.figure(figsize=(11, 3))
    randI = []
    k = 0
    for i in range(100):
        r = random.randint(0, 164)
        if r not in randI:
            randI.append(r)
    randI.sort()
    for i in range(1, 61):
        plt.subplot(4, 15, i)
        plt.imshow(image_class[randI[k]], cmap='gray')
        plt.title(label_name[randI[k]])
        plt.axis('off')
        plt.tight_layout()
        k += 1
    plt.show()


# Detect Faces and Cropping
def crop_face(image, name, edge_name):
    crop_image = []
    for i, img in enumerate(image):
        img = detect_face(img, i)
        if img is not None:
            if edge_name == 'sobel':
                Edge_Image = Sobel(img)
                crop_image.append(Edge_Image)
            elif edge_name == 'prewitt':
                Edge_Image = Prewitt(img)
                crop_image.append(Edge_Image)
            elif edge_name == 'robert':
                Edge_Image = Robert(img)
                crop_image.append(Edge_Image)
            elif edge_name == 'LBP':
                crop_image.append(img)
        else:
            del name[i]
    return crop_image


# Face Detector
def detect_face(im, idx):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.05, 3)
    try:
        x, y, w, h = faces[0]
        im = im[y:y + h, x:x + w]
        im = cv2.resize(im, (100, 100))
    except:
        print("face not found in index: ", idx)
        im = None
    return im


# Sobel Edge Detector
def Sobel(image):
    h = image.shape[0]
    w = image.shape[1]
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    SobelXImage = np.zeros((h, w))
    SobelYImage = np.zeros((h, w))
    SobelGradImage = np.zeros((h, w))
    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * image[i - 1, j - 1]) + \
                             (horizontal[0, 1] * image[i - 1, j]) + \
                             (horizontal[0, 2] * image[i - 1, j + 1]) + \
                             (horizontal[1, 0] * image[i, j - 1]) + \
                             (horizontal[1, 1] * image[i, j]) + \
                             (horizontal[1, 2] * image[i, j + 1]) + \
                             (horizontal[2, 0] * image[i + 1, j - 1]) + \
                             (horizontal[2, 1] * image[i + 1, j]) + \
                             (horizontal[2, 2] * image[i + 1, j + 1])
            SobelXImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * image[i - 1, j - 1]) + \
                           (vertical[0, 1] * image[i - 1, j]) + \
                           (vertical[0, 2] * image[i - 1, j + 1]) + \
                           (vertical[1, 0] * image[i, j - 1]) + \
                           (vertical[1, 1] * image[i, j]) + \
                           (vertical[1, 2] * image[i, j + 1]) + \
                           (vertical[2, 0] * image[i + 1, j - 1]) + \
                           (vertical[2, 1] * image[i + 1, j]) + \
                           (vertical[2, 2] * image[i + 1, j + 1])

            SobelYImage[i - 1, j - 1] = abs(verticalGrad)
            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            SobelGradImage[i - 1, j - 1] = min(255, mag)
    return SobelGradImage


# Prewitt Edge Detector
def Prewitt(image):
    h = image.shape[0]
    w = image.shape[1]
    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    PrewittXImage = np.zeros((h, w))
    PrewittYImage = np.zeros((h, w))
    PrewittGradImage = np.zeros((h, w))
    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * image[i - 1, j - 1]) + \
                             (horizontal[0, 1] * image[i - 1, j]) + \
                             (horizontal[0, 2] * image[i - 1, j + 1]) + \
                             (horizontal[1, 0] * image[i, j - 1]) + \
                             (horizontal[1, 1] * image[i, j]) + \
                             (horizontal[1, 2] * image[i, j + 1]) + \
                             (horizontal[2, 0] * image[i + 1, j - 1]) + \
                             (horizontal[2, 1] * image[i + 1, j]) + \
                             (horizontal[2, 2] * image[i + 1, j + 1])

            PrewittXImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * image[i - 1, j - 1]) + \
                           (vertical[0, 1] * image[i - 1, j]) + \
                           (vertical[0, 2] * image[i - 1, j + 1]) + \
                           (vertical[1, 0] * image[i, j - 1]) + \
                           (vertical[1, 1] * image[i, j]) + \
                           (vertical[1, 2] * image[i, j + 1]) + \
                           (vertical[2, 0] * image[i + 1, j - 1]) + \
                           (vertical[2, 1] * image[i + 1, j]) + \
                           (vertical[2, 2] * image[i + 1, j + 1])

            PrewittYImage[i - 1, j - 1] = abs(verticalGrad)

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            PrewittGradImage[i - 1, j - 1] = min(255, mag)

    return PrewittGradImage


# Robert Edge Detector
def Robert(image):
    h = image.shape[0]
    w = image.shape[1]
    horizontal = np.array([[1, 0], [0, -1]])
    vertical = np.array([[0, -1], [1, 0]])
    RobertXImage = np.zeros((h, w))
    RobertYImage = np.zeros((h, w))
    RobertGradImage = np.zeros((h, w))
    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * image[i - 1, j - 1]) + \
                             (horizontal[0, 1] * image[i - 1, j]) + \
                             (horizontal[1, 0] * image[i - 1, j + 1]) + \
                             (horizontal[1, 1] * image[i, j - 1])

            RobertXImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * image[i - 1, j - 1]) + \
                           (vertical[0, 1] * image[i - 1, j]) + \
                           (vertical[1, 0] * image[i - 1, j + 1]) + \
                           (vertical[1, 1] * image[i, j - 1])

            RobertYImage[i - 1, j - 1] = abs(verticalGrad)

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            RobertGradImage[i - 1, j - 1] = min(255, mag)

    return RobertGradImage


# Local Binary Pattern Generator
def LocalBinaryPattern(image):
    return local_binary_pattern(image, 8, 1, method="default")


# Predict Image with Model
def PredictImage(edgeType, imArr):
    match edgeType:
        case "sobel-lbp":
            lbph_SBL = cv2.face.LBPHFaceRecognizer_create()
            lbph_SBL.read("Sobel_LBPH_Model.yml")
            ID, conf = lbph_SBL.predict(imArr)
            return ID, conf
        case "prewitt-lbp":
            lbph_PRT = cv2.face.LBPHFaceRecognizer_create()
            lbph_PRT.read("Robert_LBPH_Model.yml")
            ID, conf = lbph_PRT.predict(imArr)
            return ID, conf
        case "robert-lbp":
            lbph_RBT = cv2.face.LBPHFaceRecognizer_create()
            lbph_RBT.read("Robert_LBPH_Model.yml")
            ID, conf = lbph_RBT.predict(imArr)
            return ID, conf


# Save Data Test
def save_test_image(image, name, meth):
    k = 1
    for i, img in enumerate(image):
        if meth == 'LBP':
            filenames = 'TestImage/OriginalTestImage/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1
        elif meth == 'Sobel':
            filenames = 'TestImage/Sobel/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1
        elif meth == 'Prewitt':
            filenames = 'TestImage/Prewitt/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1
        elif meth == 'Robert':
            filenames = 'TestImage/Robert/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1


# Save Data Train
def save_train_image(image, name, meth):
    k = 1
    for i, img in enumerate(image):
        if meth == 'LBP':
            filenames = 'TrainImage/OriginalTrainImage/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1
        elif meth == 'Sobel':
            filenames = 'TrainImage/SobelTrain/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1
        elif meth == 'Prewitt':
            filenames = 'TrainImage/PrewittTrain/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1
        elif meth == 'Robert':
            filenames = 'TrainImage/RobertTrain/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
            print(filenames)
            cv2.imwrite(filenames, image[i])
            k += 1



