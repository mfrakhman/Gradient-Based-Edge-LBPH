# import os
# import random
# import cv2
# import numpy as np
# import itertools
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# from os import listdir
# from yellowbrick.classifier import ClassificationReport
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from skimage.feature import local_binary_pattern
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import ConfusionMatrixDisplay
#
#
# # Showing Data
# def show_dataset(image_class, label_name):
#     plt.figure(figsize=(11, 3))
#     randI = []
#     k = 0
#     for i in range(100):
#         r = random.randint(0, 164)
#         if r not in randI:
#             randI.append(r)
#     randI.sort()
#     for i in range(1, 61):
#         plt.subplot(4, 15, i)
#         plt.imshow(image_class[randI[k]], cmap='gray')
#         plt.title(label_name[randI[k]])
#         plt.axis('off')
#         plt.tight_layout()
#         k += 1
#     plt.show()
#
#
# # Face Detector
# def detect_face(im, idx):
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(im, 1.05, 3)
#     try:
#         x, y, w, h = faces[0]
#         im = im[y:y + h, x:x + w]
#         im = cv2.resize(im, (100, 100))
#     except:
#         print("face not found in index: ", idx)
#         im = None
#     return im
#
#
# def MedianBlur(image):
#     medianKernel = (5, 5)
#     image_MedBlur = cv2.blur(image, medianKernel)
#     cv2.imwrite('assets/Face Blur/FaceBlur.jpg', image_MedBlur)
#
#     return image_MedBlur
#
#
# # Sobel Edge Detector
# def Sobel(image):
#     # image = GaussianBlur(image)
#     h = image.shape[0]
#     w = image.shape[1]
#     horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     SobelXImage = np.zeros((h, w))
#     SobelYImage = np.zeros((h, w))
#     SobelGradImage = np.zeros((h, w))
#     # offset by 1
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             horizontalGrad = (horizontal[0, 0] * image[i - 1, j - 1]) + \
#                              (horizontal[0, 1] * image[i - 1, j]) + \
#                              (horizontal[0, 2] * image[i - 1, j + 1]) + \
#                              (horizontal[1, 0] * image[i, j - 1]) + \
#                              (horizontal[1, 1] * image[i, j]) + \
#                              (horizontal[1, 2] * image[i, j + 1]) + \
#                              (horizontal[2, 0] * image[i + 1, j - 1]) + \
#                              (horizontal[2, 1] * image[i + 1, j]) + \
#                              (horizontal[2, 2] * image[i + 1, j + 1])
#             SobelXImage[i - 1, j - 1] = abs(horizontalGrad)
#
#             verticalGrad = (vertical[0, 0] * image[i - 1, j - 1]) + \
#                            (vertical[0, 1] * image[i - 1, j]) + \
#                            (vertical[0, 2] * image[i - 1, j + 1]) + \
#                            (vertical[1, 0] * image[i, j - 1]) + \
#                            (vertical[1, 1] * image[i, j]) + \
#                            (vertical[1, 2] * image[i, j + 1]) + \
#                            (vertical[2, 0] * image[i + 1, j - 1]) + \
#                            (vertical[2, 1] * image[i + 1, j]) + \
#                            (vertical[2, 2] * image[i + 1, j + 1])
#
#             SobelYImage[i - 1, j - 1] = abs(verticalGrad)
#             # Edge Magnitude
#             mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
#             SobelGradImage[i - 1, j - 1] = min(255, mag)
#
#     return SobelGradImage
#
#
# # Prewitt Edge Detector
# def Prewitt(image):
#     # image = GaussianBlur(image)
#     h = image.shape[0]
#     w = image.shape[1]
#     horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
#     vertical = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
#     PrewittXImage = np.zeros((h, w))
#     PrewittYImage = np.zeros((h, w))
#     PrewittGradImage = np.zeros((h, w))
#     # offset by 1
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             horizontalGrad = (horizontal[0, 0] * image[i - 1, j - 1]) + \
#                              (horizontal[0, 1] * image[i - 1, j]) + \
#                              (horizontal[0, 2] * image[i - 1, j + 1]) + \
#                              (horizontal[1, 0] * image[i, j - 1]) + \
#                              (horizontal[1, 1] * image[i, j]) + \
#                              (horizontal[1, 2] * image[i, j + 1]) + \
#                              (horizontal[2, 0] * image[i + 1, j - 1]) + \
#                              (horizontal[2, 1] * image[i + 1, j]) + \
#                              (horizontal[2, 2] * image[i + 1, j + 1])
#
#             PrewittXImage[i - 1, j - 1] = abs(horizontalGrad)
#
#             verticalGrad = (vertical[0, 0] * image[i - 1, j - 1]) + \
#                            (vertical[0, 1] * image[i - 1, j]) + \
#                            (vertical[0, 2] * image[i - 1, j + 1]) + \
#                            (vertical[1, 0] * image[i, j - 1]) + \
#                            (vertical[1, 1] * image[i, j]) + \
#                            (vertical[1, 2] * image[i, j + 1]) + \
#                            (vertical[2, 0] * image[i + 1, j - 1]) + \
#                            (vertical[2, 1] * image[i + 1, j]) + \
#                            (vertical[2, 2] * image[i + 1, j + 1])
#
#             PrewittYImage[i - 1, j - 1] = abs(verticalGrad)
#
#             # Edge Magnitude
#             mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
#             PrewittGradImage[i - 1, j - 1] = min(255, mag)
#
#     return PrewittGradImage
#
#
# # Robert Edge Detector
# def Robert(image):
#     # image = GaussianBlur(image)
#     h = image.shape[0]
#     w = image.shape[1]
#     horizontal = np.array([[1, 0], [0, -1]])
#     vertical = np.array([[0, -1], [1, 0]])
#     RobertXImage = np.zeros((h, w))
#     RobertYImage = np.zeros((h, w))
#     RobertGradImage = np.zeros((h, w))
#     # offset by 1
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             horizontalGrad = (horizontal[0, 0] * image[i - 1, j - 1]) + \
#                              (horizontal[0, 1] * image[i - 1, j]) + \
#                              (horizontal[1, 0] * image[i - 1, j + 1]) + \
#                              (horizontal[1, 1] * image[i, j - 1])
#
#             RobertXImage[i - 1, j - 1] = abs(horizontalGrad)
#
#             verticalGrad = (vertical[0, 0] * image[i - 1, j - 1]) + \
#                            (vertical[0, 1] * image[i - 1, j]) + \
#                            (vertical[1, 0] * image[i - 1, j + 1]) + \
#                            (vertical[1, 1] * image[i, j - 1])
#
#             RobertYImage[i - 1, j - 1] = abs(verticalGrad)
#
#             # Edge Magnitude
#             mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
#             RobertGradImage[i - 1, j - 1] = min(255, mag)
#
#     return RobertGradImage
#
#
# # Detect Faces and Cropping
# def crop_face(image, name, edge_name):
#     crop_image = []
#     for i, img in enumerate(image):
#         img = detect_face(img, i)
#         if img is not None:
#             if edge_name == 'sobel':
#                 # Blur_Image = MedianBlur(img)
#                 Edge_Image = Sobel(img)
#                 crop_image.append(Edge_Image)
#             elif edge_name == 'prewitt':
#                 # Blur_Image = MedianBlur(img)
#                 Edge_Image = Prewitt(img)
#                 crop_image.append(Edge_Image)
#             elif edge_name == 'robert':
#                 # Blur_Image = MedianBlur(img)
#                 Edge_Image = Robert(img)
#                 crop_image.append(Edge_Image)
#             elif edge_name == 'LBP':
#                 crop_image.append(img)
#         else:
#             del name[i]
#     return crop_image
#
#
# DIRECTORY = "YaleDataSet/"
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# # Gradient Edge Detection & LBPH
# P = 8
# R = 1
# img_Temp = cv2.imread("YaleDataSet/subject01/subject01.sad.png")
# img_Temp = cv2.cvtColor(img_Temp, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(img_Temp, 1.05, 3)
# try:
#     x, y, w, h = faces[0]
#     img_face = img_Temp[y:y + h, x:x + w]
#     img_face = cv2.resize(img_face, (100, 100))
# except:
#     print("face not found")
#     img_face = None
# Sobel_image = Sobel(img_face)
# Prewitt_image = Prewitt(img_face)
# Robert_image = Robert(img_face)
# LBP_image = local_binary_pattern(img_face, P=P, R=R, method="default")
# H = np.histogram(LBP_image.ravel(), bins=2 ** P, range=(0, 2 ** P), density=True)[0]
# LBP_Sobel = local_binary_pattern(Sobel_image, P, R, method="default")
# LBP_Prewitt = local_binary_pattern(Prewitt_image, P, R, method="default")
# LBP_Robert = local_binary_pattern(Robert_image, P, R, method="default")
#
# fig = plt.figure()
# ax1 = fig.add_subplot(331)
# plt.grid(False)
# plt.delaxes()
# ax2 = fig.add_subplot(332)
# plt.grid(False)
# plt.imshow(img_face, cmap="gray")
# plt.title('Face Detection')
# ax3 = fig.add_subplot(333)
# plt.grid(False)
# plt.delaxes()
# ax4 = fig.add_subplot(334)
# plt.grid(False)
# plt.imshow(Sobel_image, cmap="gray")
# plt.title('Sobel Edge')
# ax5 = fig.add_subplot(335)
# plt.grid(False)
# plt.imshow(Prewitt_image, cmap="gray")
# plt.title('Prewitt Edge')
# ax6 = fig.add_subplot(336)
# plt.grid(False)
# plt.imshow(Robert_image, cmap="gray")
# plt.title('Robert Edge')
# ax7 = fig.add_subplot(337)
# plt.grid(False)
# plt.imshow(LBP_Sobel, cmap="gray")
# plt.title('LBP-Sobel')
# ax8 = fig.add_subplot(338)
# plt.grid(False)
# plt.imshow(LBP_Prewitt, cmap="gray")
# plt.title('LBP-Prewitt')
# ax9 = fig.add_subplot(339)
# plt.grid(False)
# plt.imshow(LBP_Robert, cmap="gray")
# plt.title('LBP-Robert')
# fig.tight_layout()
#
# plt.show()
#
# # Explore Dataset
# names = []
# images = []
#
# for folder in os.listdir(DIRECTORY):
#     for name in os.listdir(os.path.join(DIRECTORY, folder)):
#         if name.find(".png") > 1:
#             img = cv2.imread(os.path.join(DIRECTORY + folder, name))
#             images.append(img)
#             names.append(folder)
#
# labels = np.unique(names)
#
# print(labels)
# print(names)
#
# print("Processing LBPH")
# # for label in labels:
# #     ids = np.where(label == np.array(names))[0]
# #     images_class = images[ids[0]: ids[-1] + 1]
#
# show_dataset(images, names)
#
#
# crop_LBP = crop_face(images, names, 'LBP')
#
# # for label in labels:
# #     ids = np.where(label == np.array(names))[0]
# #     images_class = crop_LBP[ids[0]: ids[-1] + 1]
#
# show_dataset(crop_LBP, names)
#
# labelEnd = LabelEncoder()
#
# labelEnd.fit(names)
#
# print(labelEnd.classes_)
#
# nameVec = labelEnd.transform(names)
#
# print(nameVec)
#
# # Split Dataset
# PER_CLASS = 8  # 11 images (3 test & 8 train)
# NO_CLASSES = len(labels)  # Total Number of Subject
# DS_SIZE = len(names)  # Total Number of Dataset
# TEST_SIZE = 1 - (PER_CLASS * NO_CLASSES / DS_SIZE)  # Percentage of Test Dataset
#
# x_train, x_test, y_train, y_test = train_test_split(np.array(crop_LBP, dtype=np.float32), np.array(nameVec),
#                                                     test_size=TEST_SIZE, random_state=45, stratify=np.array(nameVec))
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train)
# print(y_test)
# # Create LBPH & Predict
# lbph = cv2.face.LBPHFaceRecognizer_create()
# lbph.train(x_train, y_train)
# y_predict = [lbph.predict(x)[0] for x in x_test]
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_predict)
# np.set_printoptions(precision=3)
#
# # Display Confusion Matrix
# display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
# display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
# plt.grid(False)
# display.ax_.set_title("LBPH Confusion Matrix")
# plt.show()
#
# print("== Classification Report of LBPH OpenCV ==\n")
# print(classification_report(y_test,
#                             y_predict,
#                             target_names=labels))
#
# # Sobel-LBP
# print("Processing Sobel-LBPH")
# crop_sobel = crop_face(images, names, 'sobel')
#
# for label in labels:
#     ids = np.where(label == np.array(names))[0]
#     images_class = crop_sobel[ids[0]: ids[-1] + 1]
#
# show_dataset(crop_sobel, names)
#
# labelEnd = LabelEncoder()
#
# labelEnd.fit(names)
#
# print(labelEnd.classes_)
#
# nameVec = labelEnd.transform(names)
#
# print(nameVec)
#
# # Split Dataset
# PER_CLASS = 8  # 11 images (3 test & 8 train)
# NO_CLASSES = len(labels)  # Total Number of Subject
# DS_SIZE = len(names)  # Total Number of Dataset
# TEST_SIZE = 1 - (PER_CLASS * NO_CLASSES / DS_SIZE)  # Percentage of Test Dataset
#
# x_train, x_test, y_train, y_test = train_test_split(np.array(crop_sobel, dtype=np.float32), np.array(nameVec),
#                                                     test_size=TEST_SIZE, random_state=45, stratify=np.array(nameVec))
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#
# lbph = cv2.face.LBPHFaceRecognizer_create()
# lbph.train(x_train, y_train)
# y_predict = [lbph.predict(x)[0] for x in x_test]
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_predict)
# np.set_printoptions(precision=3)
#
# # Display Confusion Matrix
# display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
# display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
# plt.grid(False)
# display.ax_.set_title("LBP-Sobel Confusion Matrix")
# plt.show()
#
# print("== Classification Report of LBPH-Sobel OpenCV ==\n")
# print(classification_report(y_test,
#                             y_predict,
#                             target_names=labels))
#
# # Prewitt-LBP
# print("Processing Prewitt-LBPH")
# crop_prewitt = crop_face(images, names, 'prewitt')
#
# for label in labels:
#     ids = np.where(label == np.array(names))[0]
#     images_class = crop_sobel[ids[0]: ids[-1] + 1]
#
# show_dataset(crop_prewitt, names)
#
# labelEnd = LabelEncoder()
#
# labelEnd.fit(names)
#
# print(labelEnd.classes_)
#
# nameVec = labelEnd.transform(names)
#
# print(nameVec)
#
# x_train, x_test, y_train, y_test = train_test_split(np.array(crop_prewitt, dtype=np.float32), np.array(nameVec),
#                                                     test_size=TEST_SIZE, random_state=45, stratify=np.array(nameVec))
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#
# lbph = cv2.face.LBPHFaceRecognizer_create()
# lbph.train(x_train, y_train)
# y_predict = [lbph.predict(x)[0] for x in x_test]
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_predict)
# np.set_printoptions(precision=4)
#
# # Display Confusion Matrix
#
# display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
# display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
# plt.grid(False)
# display.ax_.set_title("LBP-Prewitt Confusion Matrix")
# plt.show()
#
# print("== Classification Report of LBPH-Prewitt OpenCV ==\n")
# print(classification_report(y_test,
#                             y_predict,
#                             target_names=labels))
#
# # Robert-LBP
# print("Processing Robert-LBPH")
# crop_robert = crop_face(images, names, 'robert')
#
# for label in labels:
#     ids = np.where(label == np.array(names))[0]
#     images_class = crop_robert[ids[0]: ids[-1] + 1]
#
# show_dataset(crop_robert, names)
#
# labelEnd = LabelEncoder()
#
# labelEnd.fit(names)
#
# print(labelEnd.classes_)
#
# nameVec = labelEnd.transform(names)
#
# print(nameVec)
#
# x_train, x_test, y_train, y_test = train_test_split(np.array(crop_robert, dtype=np.float32), np.array(nameVec),
#                                                     test_size=TEST_SIZE, random_state=45, stratify=np.array(nameVec))
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#
# lbph = cv2.face.LBPHFaceRecognizer_create()
# lbph.train(x_train, y_train)
# y_predict = [lbph.predict(x)[0] for x in x_test]
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_predict)
# np.set_printoptions(precision=4)
#
# # Display Confusion Matrix
#
# display = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)
# display.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
# plt.grid(False)
# display.ax_.set_title("LBP-Robert Confusion Matrix")
# plt.show()
#
# print("== Classification Report of LBPH-Robert OpenCV ==\n")
# print(classification_report(y_test,
#                             y_predict,
#                             target_names=labels))
