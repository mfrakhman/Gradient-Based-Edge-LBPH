# import cv2
# import numpy as np
# import random
# import os
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from skimage.feature import local_binary_pattern
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import ConfusionMatrixDisplay
# from tkinter import *
# from tkinter import filedialog
# from PIL import Image, ImageTk
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
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
# # Sobel Edge Detector
# def Sobel(image):
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
#     return SobelGradImage
#
#
# # Prewitt Edge Detector
# def Prewitt(image):
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
#                 Edge_Image = Sobel(img)
#                 crop_image.append(Edge_Image)
#             elif edge_name == 'prewitt':
#                 Edge_Image = Prewitt(img)
#                 crop_image.append(Edge_Image)
#             elif edge_name == 'robert':
#                 Edge_Image = Robert(img)
#                 crop_image.append(Edge_Image)
#             elif edge_name == 'LBP':
#                 crop_image.append(img)
#         else:
#             del name[i]
#     return crop_image
#
#
# def save_test_image(image, name, meth):
#     k = 1
#     for i, img in enumerate(image):
#         if meth == 'LBP':
#             filenames = 'TestImage/OriginalTestImage/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#         elif meth == 'Sobel':
#             filenames = 'TestImage/Sobel/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#         elif meth == 'Prewitt':
#             filenames = 'TestImage/Prewitt/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#         elif meth == 'Robert':
#             filenames = 'TestImage/Robert/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#
#
# def save_train_image(image, name, meth):
#     k = 1
#     for i, img in enumerate(image):
#         if meth == 'LBP':
#             filenames = 'TrainImage/OriginalTrainImage/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#         elif meth == 'Sobel':
#             filenames = 'TrainImage/SobelTrain/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#         elif meth == 'Prewitt':
#             filenames = 'TrainImage/PrewittTrain/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#         elif meth == 'Robert':
#             filenames = 'TrainImage/RobertTrain/' + str(k) + '.Subject' + str(name[i] + 1) + '.png'
#             print(filenames)
#             cv2.imwrite(filenames, image[i])
#             k += 1
#
#
# def showImage():
#     filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",
#                                           filetype=(("PNG File", "*.png"), ("JPG File", "*.jpg")))
#     LabelSubject.configure(text="Subject: " + filename[-6:])
#     img = Image.open(filename)
#     imArr = np.array(img)
#     # Select Image
#     img = ImageTk.PhotoImage(img)
#     LB1.configure(text="Image Selected", image=img, compound='top')
#     LB1.image = img
#
#     # Sobel Edge
#     img_sobel = Sobel(imArr)
#     sobArr = np.array(img_sobel)
#     img_sobel = ImageTk.PhotoImage(Image.fromarray(img_sobel))
#     LB2.configure(image=img_sobel)
#     LB2.image = img_sobel
#     # Prewitt Edge
#     img_prewitt = Prewitt(imArr)
#     preArr = np.array(img_prewitt)
#     img_prewitt = ImageTk.PhotoImage(Image.fromarray(img_prewitt))
#     LB3.configure(image=img_prewitt)
#     LB3.image = img_prewitt
#     # Robert Edge
#     img_robert = Robert(imArr)
#     robArr = np.array(img_robert)
#     img_robert = ImageTk.PhotoImage(Image.fromarray(img_robert))
#     LB4.configure(image=img_robert)
#     LB4.image = img_robert
#     # Show LBP Operation
#     img_sobLBP = local_binary_pattern(sobArr, 8, 1, method="default")
#
#     img_sobLBP = ImageTk.PhotoImage(Image.fromarray(img_sobLBP))
#     LB5.configure(image=img_sobLBP)
#     LB5.image = img_sobLBP
#     img_preLBP = local_binary_pattern(preArr, 8, 1, method='default')
#     img_preLBP = ImageTk.PhotoImage(Image.fromarray(img_preLBP))
#     LB6.configure(image=img_preLBP)
#     LB6.image = img_preLBP
#     img_robLBP = local_binary_pattern(robArr, 8, 1, method='default')
#     img_robLBP = ImageTk.PhotoImage(Image.fromarray(img_robLBP))
#     LB7.configure(image=img_robLBP)
#     LB7.image = img_robLBP
#
#     lbph_SBL = cv2.face.LBPHFaceRecognizer_create()
#     lbph_SBL.read("Sobel_LBPH_Model.yml")
#     imgS = sobArr
#     ID, conf = lbph_SBL.predict(imgS)
#     LB8.configure(text="Subject Predicted: " + str(ID + 1) + "\n Euclidean Calculation: " + str(conf))
#     lbph_PRT = cv2.face.LBPHFaceRecognizer_create()
#     lbph_PRT.read("Prewitt_LBPH_Model.yml")
#     imgP = preArr
#     ID, conf = lbph_PRT.predict(imgP)
#     LB9.configure(text="Subject Predicted: " + str(ID + 1) + "\n Euclidean Calculation: " + str(conf))
#     lbph_RBT = cv2.face.LBPHFaceRecognizer_create()
#     lbph_RBT.read("Robert_LBPH_Model.yml")
#     imgR = robArr
#     ID, conf = lbph_RBT.predict(imgR)
#     LB10.configure(text="Subject Predicted: " + str(ID + 1) + "\n Euclidean Calculation: " + str(conf))
#
#
# # def predict_test():
# #     top = Toplevel()
# #     top.title("Prediction on Each Model")
# #     imageS = LB5.cget('image')
# #     imageS_arr = ImageTk.PhotoImage(np.array(imageS))
# #     print(imageS_arr)
# #     sobel_lbp = Label(top, text="Sobel-LBP Image", image=imageS, compound='top')
# #     sobel_lbp.pack()
# #     prewitt_lbp = Label(top, text="Prewitt-LBP Image")
# #     prewitt_lbp.pack()
# #     robert_lbp = Label(top, text="Robert-LBP Image")
# #     robert_lbp.pack()
# #     prediction1 = Label(top, text="Prediction 1")
# #     prediction1.pack()
#
#
# def main():
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("Program is finished")
#
#
# if __name__ == '__main__':
#     main()
#
# root = Tk()
# root.title("Gradient Edge - LBP")
# root.geometry("900x500")
#
# root.grid_rowconfigure(0, weight=0)
# root.grid_columnconfigure(0, weight=1)
#
# F1 = Frame(root, width=100, height=100)
# F1.grid(row=0, column=0, sticky="ns")
#
# F2 = Frame(root, width=100, height=100)
# F2.grid(row=1, column=0, sticky="ns")
#
# F3 = Frame(root, width=100, height=100)
# F3.grid(row=2, column=0, sticky="ns")
#
# F4 = Frame(root, width=100, height=100)
# F4.grid(row=3, column=0, sticky="ns")
#
# F5 = Frame(root, width=100, height=100)
# F5.grid(row=4, column=0, sticky="ns")
#
# LB1 = Label(F1, text="Test Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB1.grid(row=0, column=0, padx=10, pady=10)
#
# LabelSubject = Label(F1, text="Subject Number", padx=10, pady=10)
# LabelSubject.grid(row=0, column=1, padx=10, pady=10)
#
# LB2 = Label(F2, text="Sobel Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB2.grid(row=0, column=0, padx=10, pady=10)
#
# LB3 = Label(F2, text="Prewitt Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB3.grid(row=0, column=1, padx=10, pady=10)
#
# LB4 = Label(F2, text="Robert Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB4.grid(row=0, column=2, padx=10, pady=10)
#
# LB5 = Label(F3, text="Sobel-LBP Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB5.grid(row=0, column=0, padx=10, pady=10)
#
# LB6 = Label(F3, text="Prewitt-LBP Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB6.grid(row=0, column=1, padx=10, pady=10)
#
# LB7 = Label(F3, text="Robert-LBP Image", borderwidth=1, relief="solid", padx=10, pady=10)
# LB7.grid(row=0, column=2, padx=10, pady=10)
#
# LB8 = Label(F4, text="Prediction Sobel-LBP", borderwidth=1, relief="solid", padx=10, pady=10)
# LB8.grid(row=0, column=0, padx=10, pady=10)
#
# LB9 = Label(F4, text="Prediction Prewitt-LBP", borderwidth=1, relief="solid", padx=10, pady=10)
# LB9.grid(row=0, column=1, padx=10, pady=10)
#
# LB10 = Label(F4, text="Prediction Robert-LBP", borderwidth=1, relief="solid", padx=10, pady=10)
# LB10.grid(row=0, column=2, padx=10, pady=10)
#
# BTN1 = Button(F5, text="Select Image", command=showImage)
# BTN1.grid(row=0, column=0)
#
# BTN2 = Button(F5, text="Predict")
# BTN2.grid(row=0, column=1)
#
# root.mainloop()
