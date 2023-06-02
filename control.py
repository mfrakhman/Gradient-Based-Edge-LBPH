import cv2
import numpy as np
import os
from processing import crop_face, save_train_image, save_test_image, Sobel, Prewitt, Robert, LocalBinaryPattern, PredictImage
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def model_and_testLBPH():
    # LBPH
    print("Starting Classification LBPH")
    crop_LBP = crop_face(images, names, 'LBP')
    # show_dataset(crop_LBP, names)
    x_train, x_test, y_train, y_test = train_test_split(np.array(crop_LBP, dtype=np.float32), np.array(nameVec),
                                                        test_size=TEST_SIZE, random_state=45,
                                                        stratify=np.array(nameVec))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    save_test_image(x_test, y_test, 'LBP')
    save_train_image(x_train, y_train, 'LBP')
    # Create and Save LBPH Train Model
    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.train(x_train, y_train)
    lbph.write('LBPH_Model.yml')
    # Predict LBPH Test Image Set
    y_predict = [lbph.predict(x)[0] for x in x_test]  # K-NN Classifier with Chi-Square Dist Calculation
    # Create and Display Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=3)
    display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
    display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    display.ax_.set_title("LBPH Confusion Matrix")
    # plt.show()
    print("== Classification Report of LBPH OpenCV ==\n")
    print(classification_report(y_test,
                                y_predict,
                                target_names=labels))
    plt.show(block=True)


def model_and_testSobel():
    # Sobel-LBPH
    print("Starting Classification Sobel-LBPH")
    # Face Detect, Crop, and Edge (Optional)
    crop_sobel = crop_face(images, names, 'sobel')
    # show_dataset(crop_sobel, names)
    # Split Dataset
    x_train, x_test, y_train, y_test = train_test_split(np.array(crop_sobel, dtype=np.float32), np.array(nameVec),
                                                        test_size=TEST_SIZE, random_state=45,
                                                        stratify=np.array(nameVec))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Save Test Image Set
    save_test_image(x_test, y_test, 'Sobel')
    save_train_image(x_train, y_train, 'Sobel')
    # Create and Save LBPH Train Model
    sobel_lbph = cv2.face.LBPHFaceRecognizer_create()
    sobel_lbph.train(x_train, y_train)
    sobel_lbph.write('Sobel_LBPH_Model.yml')
    # Predict Sobel-LBPH Test Image Set
    y_predict = [sobel_lbph.predict(x)[0] for x in x_test]  # K-NN Classifier with Chi-Square Dist Calculation
    # Create and Display confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=3)
    display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
    display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    plt.grid(False)
    display.ax_.set_title("Sobel-LBPH Confusion Matrix")
    plt.savefig('Sobel-LBPH_ConfusionMatrix.png')
    # plt.show()
    print("== Classification Report of Sobel-LBPH OpenCV ==\n")
    print(classification_report(y_test,
                                y_predict,
                                target_names=labels))
    plt.show(block=True)


def model_and_testPrewitt():
    # Prewitt-LBPH
    print("Starting Classification Prewitt-LBPH")
    # Face Detect, Crop, and Edge (Optional)
    crop_prewitt = crop_face(images, names, 'prewitt')
    # show_dataset(crop_prewitt, names)
    # Split Dataset
    x_train, x_test, y_train, y_test = train_test_split(np.array(crop_prewitt, dtype=np.float32), np.array(nameVec),
                                                        test_size=TEST_SIZE, random_state=45,
                                                        stratify=np.array(nameVec))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Save Test Image Set
    save_test_image(x_test, y_test, 'Prewitt')
    save_train_image(x_train, y_train, 'Prewitt')
    # Create and Save Prewitt-LBPH Train Model
    prewitt_lbph = cv2.face.LBPHFaceRecognizer_create()
    prewitt_lbph.train(x_train, y_train)
    prewitt_lbph.write('Prewitt_LBPH_Model.yml')
    # Predict Prewitt-LBPH Test Image Set
    y_predict = [prewitt_lbph.predict(x)[0] for x in x_test]  # K-NN Classifier with Chi-Square Dist Calculation
    # Create and Display confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=3)
    display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
    display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    plt.grid(False)
    display.ax_.set_title("Prewitt-LBPH Confusion Matrix")
    plt.savefig('Prewitt-LBPH_ConfusionMatrix.png')
    # plt.show()
    print("== Classification Report of Prewitt-LBPH OpenCV ==\n")
    print(classification_report(y_test,
                                y_predict,
                                target_names=labels))
    plt.show(block=True)


def model_and_testRobert():
    # Robert-LBPH
    print("Starting Classification Robert-LBPH")
    # Face Detect, Crop, and Edge (Optional)
    crop_robert = crop_face(images, names, 'robert')
    # show_dataset(crop_robert, names)
    x_train, x_test, y_train, y_test = train_test_split(np.array(crop_robert, dtype=np.float32), np.array(nameVec),
                                                        test_size=TEST_SIZE, random_state=45,
                                                        stratify=np.array(nameVec))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Save Test Image Set
    save_test_image(x_test, y_test, 'Robert')
    save_train_image(x_train, y_train, 'Robert')

    # Create and Save Robert-LBPH Train Model
    robert_lbph = cv2.face.LBPHFaceRecognizer_create()
    robert_lbph.train(x_train, y_train)
    robert_lbph.write('Robert_LBPH_Model.yml')
    # Predict Robert-LBPH Test Image Set
    y_predict = [robert_lbph.predict(x)[0] for x in x_test]  # K-NN Classifier with Chi-Square Dist Calculation
    # Create and Display confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=3)
    display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
    display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    plt.grid(False)
    display.ax_.set_title("Robert-LBPH Confusion Matrix")
    plt.savefig('Robert-LBPH_ConfusionMatrix.png')
    # plt.show()
    print("== Classification Report of Robert-LBPH OpenCV ==\n")
    print(classification_report(y_test,
                                y_predict,
                                target_names=labels))
    plt.show(block=True)


def edgeImage(imArr, edgeType):
    match edgeType:
        case "sobel":
            return Sobel(imArr)
        case "prewitt":
            return Prewitt(imArr)
        case "robert":
            return Robert(imArr)


def lbpImage(imArr):
    return LocalBinaryPattern(imArr)


def seePrediction(edgeType, imArr):
    match edgeType:
        case "sobel-lbp":
            return PredictImage(edgeType, imArr)
        case "prewitt-lbp":
            return PredictImage(edgeType, imArr)
        case "robert-lbp":
            return PredictImage(edgeType, imArr)


# Read Dataset
DIRECTORY = "YaleDataSet/"
names = []
images = []
for folder in os.listdir(DIRECTORY):
    for name in os.listdir(os.path.join(DIRECTORY, folder)):
        if name.find(".png") > 1:
            img = cv2.imread(os.path.join(DIRECTORY + folder, name))
            images.append(img)
            names.append(folder)

# Labeling Dataset
labels = np.unique(names)
labelEnd = LabelEncoder()
labelEnd.fit(names)
print(labelEnd.classes_)
nameVec = labelEnd.transform(names)
print(nameVec)

# Size of the Train and Testing Data
TOTAL_TRAIN = 8  # 11 images (3 test & 8 train)
TOTAL_SUBJECT = len(labels)  # Total Number of Subject
DATASET_SIZE = len(names)  # Total Number of Dataset
TEST_SIZE = 1 - (TOTAL_TRAIN * TOTAL_SUBJECT / DATASET_SIZE)  # Percentage of Test Dataset

