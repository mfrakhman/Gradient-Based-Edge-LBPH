import os
import numpy as np
from control import model_and_testLBPH, model_and_testSobel, model_and_testPrewitt, model_and_testRobert, edgeImage, lbpImage, seePrediction
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


imArr, sobArr, preArr, robArr = [], [], [], []

# Create an instance of tkinter frame or window
win = Tk()

# Set the size of the window
win.geometry("800x300")
win.title("Testing each Model")

# Create two frames in the window
button_frame = Frame(win)
frame1 = Frame(win)
frame2 = Frame(win)
frame3 = Frame(win)
frame4 = Frame(win)


# Switch Frame to Select Image Frame
def switch_frame_select_image():
    frame1.pack(fill='both', expand=1)
    frame2.pack_forget()
    frame3.pack_forget()
    frame4.pack_forget()
    btn_LBP.pack_forget()
    btn_Sobel_LBP.pack_forget()
    btn_Prewitt_LBP.pack_forget()
    btn_Robert_LBP.pack_forget()
    btn1.configure(text="Select Image")
    btn1.pack(side=LEFT, pady=20, padx=5)
    btn2.pack(side=LEFT, pady=20, padx=5)
    btn3.pack_forget()
    btn4.pack_forget()


# Switch Frame to Edge Image Frame
def switch_frame_edge_image():
    frame2.pack(fill='both', expand=1)
    frame1.pack_forget()
    frame3.pack_forget()
    frame4.pack_forget()
    btn3.pack(side=LEFT, pady=20, padx=5)
    btn4.pack(side=LEFT, pady=20, padx=5)


# Switch Frame to Edge-LBP Image Frame
def switch_frame_lbp_image():
    frame3.pack(fill='both', expand=1)
    frame1.pack_forget()
    frame2.pack_forget()
    frame4.pack_forget()


# Switch Frame to Prediction Frame
def switch_frame_prediction_image():
    frame4.pack(fill='both', expand=1)
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()


# Define a function for switching the frames
def switch_frame(frame):
    match frame:
        case 1:
            switch_frame_select_image()
        case 2:
            switch_frame_edge_image()
        case 3:
            switch_frame_lbp_image()
        case 4:
            switch_frame_prediction_image()


# Selecting Image Function
def select_image():
    switch_frame(1)
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",
                                          filetype=(("PNG File", "*.png"), ("JPG File", "*.jpg")))
    print(filename)
    img = Image.open(filename)
    Arr = np.array(img)
    global imArr
    imArr = Arr
    img = ImageTk.PhotoImage(img)
    selected_image_label.configure(text="Selected Image", image=img, compound='top')
    selected_image_label.image = img
    selected_subject_label.configure(text="Subject: " + filename)


# Edge Detection on Every Gradient Based Method
def see_edge():
    switch_frame(2)
    global imArr, sobArr, preArr, robArr
    img_sobel = edgeImage(imArr, "sobel")
    sobArr = np.array(img_sobel)
    img_sobel = ImageTk.PhotoImage(Image.fromarray(img_sobel))
    sobel_image_label.configure(text="Sobel Image", image=img_sobel, compound='top')
    sobel_image_label.image = img_sobel
    img_prewitt = edgeImage(imArr, "prewitt")
    preArr = np.array(img_prewitt)
    img_prewitt = ImageTk.PhotoImage(Image.fromarray(img_prewitt))
    prewitt_image_label.configure(text="Prewitt Image", image=img_prewitt, compound='top')
    prewitt_image_label.image = img_prewitt
    img_robert = edgeImage(imArr, "robert")
    robArr = np.array(img_robert)
    img_robert = ImageTk.PhotoImage(Image.fromarray(img_robert))
    robert_image_label.configure(text="Robert Image", image=img_robert, compound='top')
    robert_image_label.image = img_robert


# LBP Operation on Each Edge Image
def see_LBP():
    switch_frame(3)
    global sobArr, preArr, robArr
    img_sobLBP = lbpImage(sobArr)
    img_sobLBP = ImageTk.PhotoImage(Image.fromarray(img_sobLBP))
    sobel_LBP_image_label.configure(text="Sobel-LBP Image", image=img_sobLBP, compound='top')
    sobel_LBP_image_label.image = img_sobLBP
    img_preLBP = lbpImage(preArr)
    img_preLBP = ImageTk.PhotoImage(Image.fromarray(img_preLBP))
    prewitt_LBP_image_label.configure(text="Prewitt-LBP Image", image=img_preLBP, compound='top')
    prewitt_LBP_image_label.image = img_preLBP
    img_robLBP = lbpImage(robArr)
    img_robLBP = ImageTk.PhotoImage(Image.fromarray(img_robLBP))
    robert_LBP_image_label.configure(text="Robert-LBP Image", image=img_robLBP, compound='top')
    robert_LBP_image_label.image = img_robLBP


# Prediction on Each Image
def see_prediction():
    switch_frame(4)
    global sobArr, preArr, robArr
    imgS = sobArr
    ID, conf = seePrediction("sobel-lbp", imgS)
    sobel_LBP_image_prediction_label.configure(
        text="Subject Predicted: " + str(ID + 1) + "\n Chi-Square Calculation: " + str(conf))
    imgP = preArr
    ID, conf = seePrediction("prewitt-lbp", imgP)
    prewitt_LBP_image_prediction_label.configure(
        text="Subject Predicted: " + str(ID + 1) + "\n Chi-Square Calculation: " + str(conf))
    imgR = robArr
    ID, conf = seePrediction("robert-lbp", imgR)
    robert_LBP_image_prediction_label.configure(
        text="Subject Predicted: " + str(ID + 1) + "\n Chi-Square Calculation: " + str(conf))


# Selected Image Label
selected_image_label = Label(frame1)
selected_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)
selected_subject_label = Label(frame1)
selected_subject_label.pack(side=LEFT, expand=True, pady=10, padx=10)

# Edge Detection Label
sobel_image_label = Label(frame2)
sobel_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)
prewitt_image_label = Label(frame2)
prewitt_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)
robert_image_label = Label(frame2)
robert_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)

# LBP Image Label
sobel_LBP_image_label = Label(frame3)
sobel_LBP_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)
prewitt_LBP_image_label = Label(frame3)
prewitt_LBP_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)
robert_LBP_image_label = Label(frame3)
robert_LBP_image_label.pack(side=LEFT, expand=True, pady=10, padx=10)

# Prediction Label
sobel_LBP_image_prediction_label = Label(frame4)
sobel_LBP_image_prediction_label.pack(side=LEFT, expand=True, pady=10, padx=10)
prewitt_LBP_image_prediction_label = Label(frame4)
prewitt_LBP_image_prediction_label.pack(side=LEFT, expand=True, pady=10, padx=10)
robert_LBP_image_prediction_label = Label(frame4)
robert_LBP_image_prediction_label.pack(side=LEFT, expand=True, pady=10, padx=10)

# Button Selection
btn_LBP = Button(button_frame, text="LBPH", command=model_and_testLBPH)
btn_LBP.pack(side=LEFT, pady=20, padx=5)
btn_Sobel_LBP = Button(button_frame, text="Sobel-LBPH", command=model_and_testSobel)
btn_Sobel_LBP.pack(side=LEFT, pady=20, padx=5)
btn_Prewitt_LBP = Button(button_frame, text="Prewitt-LBPH", command=model_and_testPrewitt)
btn_Prewitt_LBP.pack(side=LEFT, pady=20, padx=5)
btn_Robert_LBP = Button(button_frame, text="Robert-LBPH", command=model_and_testRobert)
btn_Robert_LBP.pack(side=LEFT, pady=20, padx=5)
btn1 = Button(button_frame, text="Test Selected Image", command=select_image, activebackground='#ff0000')
btn1.pack(side=LEFT, pady=20, padx=(100, 5))
btn2 = Button(button_frame, text="Edge Detection", command=see_edge, activebackground='#00ff00')
btn3 = Button(button_frame, text="LBP Image", command=see_LBP, activebackground='#0000ff')
btn4 = Button(button_frame, text="Predict", command=see_prediction, activebackground='#ffff00')
button_frame.pack(side=BOTTOM)

win.mainloop()
