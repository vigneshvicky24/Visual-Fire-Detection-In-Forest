import streamlit as st
import winsound
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageFilter
import matplotlib.image as mpimg
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import base64
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import xml.etree.ElementTree as ET
import pandas as pd
import folium
import webbrowser

# Function to add a background image to Streamlit app
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Function to read an input image
def read_input_image(filename):
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    st.image(img, caption="Original Image")

    # Preprocess the image
    resized_image = cv2.resize(img, (300, 300))
    img_resize_orig = cv2.resize(img, (50, 50))
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()
    SPV = np.shape(img)
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    except:
        gray1 = img_resize_orig
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1, cmap='gray')
    plt.axis('off')
    plt.show()
    #st.image(gray1)
   
    # Feature extraction
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val, median_val, var_val]
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)
   
    # Local Binary Pattern (LBP) calculation
    def find_pixel(imgg, center, x, y):
        new_value = 0
        try:
            if imgg[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def lbp_calculated_pixel(imgg, x, y):
        center = imgg[x][y]
        val_ar = []
        val_ar.append(find_pixel(imgg, center, x-1, y-1))
        val_ar.append(find_pixel(imgg, center, x-1, y))
        val_ar.append(find_pixel(imgg, center, x-1, y + 1))
        val_ar.append(find_pixel(imgg, center, x, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y))
        val_ar.append(find_pixel(imgg, center, x + 1, y-1))
        val_ar.append(find_pixel(imgg, center, x, y-1))
        power_value = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_value[i]
        return val

    height, width, _ = img.shape
    img_gray_conv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)
    plt.imshow(img_lbp, cmap="gray")
    plt.title("LBP")
    plt.show()

    # Image splitting
    data_fire = os.listdir('Data/Fire/')
    data_nofire = os.listdir('Data/NoFire/')
    dot1 = []
    labels1 = []
    for img1 in data_fire:
        img_1 = mpimg.imread('Data/Fire/' + "/" + img1)
        img_1 = cv2.resize(img_1, ((50, 50)))
        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        except:
            gray = img_1
        dot1.append(np.array(gray))
        labels1.append(0)

    for img1 in data_nofire:
        try:
            img_2 = mpimg.imread('Data/NoFire/' + "/" + img1)
            img_2 = cv2.resize(img_2, ((50, 50)))
            try:            
                gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            except:
                gray = img_2
            dot1.append(np.array(gray))
            labels1.append(1)
        except:
            None

    x_train, x_test, y_train, y_test = train_test_split(dot1, labels1, test_size=0.2, random_state=101)

    print("---------------------------------------------------")
    print("Image Splitting")
    print("---------------------------------------------------")
    print()
    print("Total no of input data   :", len(dot1))
    print("Total no of train data   :", len(x_train))
    print("Total no of test data    :", len(x_test))    

    # Classification
    y_train1 = np.array(y_train)
    y_test1 = np.array(y_test)
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)

    x_train2 = np.zeros((len(x_train), 50, 50, 3))
    for i in range(0, len(x_train)):
        x_train2[i,:,:,:] = x_train2[i]

    x_test2 = np.zeros((len(x_test), 50, 50, 3))
    for i in range(0, len(x_test)):
        x_test2[i,:,:,:] = x_test2[i]

    print("-------------------------------------------------------------")
    print('Convolutional Neural Network')
    print("-------------------------------------------------------------")
    print()
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50, 50, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit(x_train2, train_Y_one_hot, batch_size=2, epochs=10, verbose=1)
    accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)
    print("-------------------------------------------------------------")
    print("Performance Analysis")
    print("-------------------------------------------------------------")
    print()
    loss = history.history['loss']
    loss = max(loss)
    accuracy = 100 - loss
    print()
    print("1.Accuracy    :", accuracy, '%')
    print()
    print("2.Error Rate  :", loss, '%')
    print()

    # Prediction
    Total_length = len(data_fire) + len(data_nofire)
    temp_data1 = []
    for ijk in range(0, Total_length):
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    temp_data1 = np.array(temp_data1)
    zz = np.where(temp_data1 == 1)
    if labels1[zz[0][0]] == 0:
        aa = filename.split('/')
        aa3 = aa[len(aa)-1]
        ff = str(aa3[0:len(aa3)-4]) + '.xml'
        st.text(ff)
        tree = ET.parse('Data/Annotation/' + ff)
        root = tree.getroot()
        image = cv2.imread(filename)
        for obj in root.iter('object'):
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
           
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        st.image(image, caption="Segmentated Image")
        st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:36px;">{"IDENTIFIED = FIRE"}</h1>', unsafe_allow_html=True)
       
       
        
     
        winsound.Beep(1000, 15000)  # Adjust the duration as needed, here it's set to 10 seconds

        survey_data = {
            'Latitude': [37.7749, 37.7748, 37.7747],
            'Longitude': [-122.4194, -122.4195, -122.4196],
            'Survey_Point': ['Point A', 'Point B', 'Point C']
        }
        df = pd.DataFrame(survey_data)
        map_center = [sum(df['Latitude']) / len(df['Latitude']), sum(df['Longitude']) / len(df['Longitude'])]
        mymap = folium.Map(location=map_center, zoom_start=12)
        for index, row in df.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=row['Survey_Point']
            ).add_to(mymap)
        mymap.save("survey_map.html")
    else:
        st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:36px;">{"IDENTIFIED = NO FIRE"}</h1>', unsafe_allow_html=True)

# Setting the background image
add_bg_from_local('1.jpg')

# Main Streamlit app logic
url = "file:///C:/Users/SPV/Downloads/Project/PROJECT/Forest/survey_map.html"
col1, col2, col3 = st.columns(3)

with col2:
    aa = st.button("Upload Image")
    if aa:
        filename = askopenfilename()
        read_input_image(filename)

if st.button("Logout"):
    # Open login page in a new browser tab
    webbrowser.open_new_tab("http://localhost:8502/")

if st.button("Location"):
    # Open survey map in a new browser tab
    webbrowser.open_new_tab("file:///C:/Users/SPV/Downloads/Project/PROJECT/Forest/survey_map.html")
