import os
import numpy as np
import cv2
#import dlib
#from imutils import face_utils
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import applications
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, LSTM, Input, Lambda, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import TimeDistributed
import sklearn
import datetime
from tqdm import tqdm
import fnmatch
from time import gmtime, strftime
from multiprocessing.dummy import Pool as ThreadPool
import itertools
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
from keras import backend as K
import sys

from sklearn.utils import shuffle
IMG_DATA = '/data/scanavan1/BP4D+/2D+3D/'
LANDMARK_DATA = '/data/scanavan1/BP4D+/3DFeatures/FacialLandmarks/'

def get_landmark_paths(path):
	pain = [] 
	no_pain = []
	p_frames = []
	np_frames = []

	if os.path.exists(path):
		files = os.walk(path).__next__()[2]
		f = [x.strip('.bndplus').split('_') for x in files]
		for idx, val in enumerate(f):
			if val[1] == 'T8':
				pain.append(files[idx])	
			else:
				no_pain.append(files[idx])	
	
	b_pain = shuffle(pain) 	
	b_no_pain = shuffle(no_pain)
	return b_pain, b_no_pain


def get_pain_landmarks(pain_paths):
	frames = []
	pain = []
	
	for v in pain_paths:
		frames.append((v.strip('.bndplus').split('_')))
		with open(LANDMARK_DATA + v) as flandmarks:
			landmarks = []
			for line in flandmarks.readlines():
				landmarks.extend(line.strip().split(','))
			pain.append([float(v) for v in landmarks])
	
	return frames, pain

def get_no_pain_landmarks(no_pain_paths):
	frames = []
	no_pain = []

	for v in no_pain_paths:
		frames.append((v.strip('.bndplus').split('_')))
		with open(LANDMARK_DATA + v) as flandmarks:
			landmarks = []
			for line in flandmarks.readlines():
				landmarks.extend(line.strip().split(','))
			no_pain.append([float(v) for v in landmarks])
        
	return frames, no_pain

def get_imgs(path, p_frames, np_frames):
	p_imgs = []
	np_imgs = []
	if os.path.exists(path):
		subjects = os.walk(path).__next__()[1]
		for subject in subjects:
			tasks = os.walk(path + subjects[0]).__next__()[1]
			for task in tasks:
				files = os.walk(path + subject + '/' + task).__next__()[2]
				imgs = [f.strip('.jpg') for f in files if f.endswith('.jpg')]
				for frame in imgs:
					if [subject, task, frame] in p_frames:
						#pain.append(path + subject + '/' + task + '/' + frame + '.jpg')
						ppath = path + subject + '/' + task + '/' + frame + '.jpg'
						pimg = cv2.imread(ppath)
						pimg_r = cv2.resize(pimg, (128, 128))
						p_imgs.append(pimg_r.shape)
					elif [subject, task, frame] in np_frames:
						#no_pain.append(path + subject + '/' + task + '/' + frame + '.jpg')
						nppath = path + subject + '/' + task + '/' + frame + '.jpg'
						npimg = cv2.imread(nppath)
						npimg_r = cv2.resize(npimg, (128, 128))
						np_imgs.append(npimg_r.shape)
	return p_imgs, np_imgs		


#detect face in image
def DetectFace(cascade, image, scale_factor=1.1):
    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          
    #find face(s) in image
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=3, minSize=(110,110))
#    rects = detector(gray, 0)
    
    outputFace = None
#    outputLandmarks = None
    #crop image to face region
    for face in faces:
        if face is None:
#            print('Face not detected in {}'.format(image))
            break
        else:
            x,y,w,h = face
            outputFace = image[y:y+h, x:x+w]
#            outputLandmarks = face_utils.shape_to_np(predictor(gray, rects[0]))
#            cv2.imwrite("Original.jpg", image)
#            cv2.imwrite("Crop.jpg", outputFace)
#            sys.exit()
#            return outputFace, outputLandmarks.flatten(), True
            return outputFace, True
    return outputFace, False

def directorySearch(directory, label, dataName, dataAugmentation=False):
#    print('Started directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    x, y = [], []
#    landmarksOutput = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    if label is 0:
#        fileBadImages = open('{0}-BadImagesNoPain.txt'.format(time), 'w+')
#        fileBadFaces = open('{0}-BadFacesNoPain.txt'.format(time), 'w+')
        pass
    elif label is 1:
#        fileBadImages = open('{0}-BadImagesPain.txt'.format(time), 'w+')
#        fileBadFaces = open('{0}-BadFacesPain.txt'.format(time), 'w+')
        pass
    else:
        print('Error: label should be 0 or 1')
        return
    countBadImages = 0
    countBadFaces = 0
#    detector = dlib.get_frontal_face_detector()
#    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    img_gen = ImageDataGenerator()
#    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))[0:100]):
    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))):
        if file.endswith('.jpg'):
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            if img is None:
#                fileBadImages.write(file + '\n')
                countBadImages += 1
                pass
            else:
                face, faceDetected = DetectFace(face_cascade, img)
                if faceDetected:
                    faceResized = cv2.resize(face, (128, 128), interpolation = cv2.INTER_AREA)
#                    print(faceResized.shape)
#                    cv2.imwrite("Original.jpg", faceResized)
                    x.append(faceResized)
                    y.append(label)
#                    landmarksOutput.append(landmarks)
                    
                    if dataAugmentation:
                        # transformation types
                        for theta_input in [-15,-10,-5,0,5,10,15]:
#                            for flip_horizontal_input in [False, True]:
                            for flip_horizontal_input in [False]:
                                for flip_vertical_input in [False, True]:
#                                    for channel_shift_intencity_input in [-100,0,100]:
                                    for channel_shift_intencity_input in [0]:
                                        faceTransform = img_gen.apply_transform(faceResized,{'theta':theta_input,'flip_horizontal':flip_horizontal_input,'flip_vertical':flip_vertical_input,'channel_shift_intencity':channel_shift_intencity_input})
                                        x.append(faceTransform)
                                        y.append(label)
#                        print(faceRotate.shape)
#                        cv2.imwrite("Rotate.jpg", faceRotate)
#                        # augmented data: rotate
#                        faceRotate = img_gen.apply_transform(faceResized, {'channel_shift_intencity':-100})
#                        print(faceRotate.shape)
#                        cv2.imwrite("Rotate.jpg", faceRotate)
#                        x.append(faceRotate)
#                        y.append(label)
                        
                        # augmented data: mirror (vertical flip)
#                        faceMirror = cv2.flip(faceResized, 1)
    #                    print(faceMirror.shape)
    #                    cv2.imwrite("Mirror.jpg", faceMirror)
#                        x.append(faceMirror)
#                        y.append(label)
                    
                        # augmented data: Gaussian Blur
#                        faceBlur = gaussian_filter(faceResized, sigma=0.5)
    #                    print(faceBlur.shape)
    #                    cv2.imwrite("Blur.jpg", faceBlur)
#                        x.append(faceBlur)
#                        y.append(label)
                        
                        # augmented data: mirror and Gaussian Blur
#                        faceBlurMirror = gaussian_filter(faceMirror, sigma=0.5)
#                        x.append(faceBlurMirror)
#                        y.append(label)
#                        return
                else:
#                    fileBadFaces.write(file + '\n')
                    countBadFaces += 1
#    if countBadImages > 0:
#        print('Bad images count: {}'.format(countBadImages))
#    if countBadFaces > 0:
#        print('Bad faces count: {}'.format(countBadFaces))
#    print('Ended directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    return x, y

def verifyLength(list1, list2, list1Name, list2Name):
    if len(list1) != len(list2):
        print('Error: {0} length does not equal {1} length'.format(list1Name, list2Name))

# preprocessing
def readImages(pathData):
    # get test data
    pathTestNoPain = '{}Testing/No_pain/'.format(pathData)
    x_TestNoPain, y_TestNoPain = directorySearch(pathTestNoPain, 0, 'Test No Pain')
    verifyLength(x_TestNoPain, y_TestNoPain, 'x_TestNoPain', 'y_TestNoPain')
    pathTestPain = '{}Testing/Pain/'.format(pathData)
    x_TestPain, y_TestPain = directorySearch(pathTestPain, 1, 'Test Pain')
    verifyLength(x_TestPain, y_TestPain, 'x_TestPain', 'y_TestPain')

    # get train data
    pathTrainNoPain = '{}Training/No_pain/'.format(pathData)
    x_TrainNoPain, y_TrainNoPain = directorySearch(pathTrainNoPain, 0, 'Train No Pain', dataAugmentation=False)
    verifyLength(x_TrainNoPain, y_TrainNoPain, 'x_TrainNoPain', 'y_TrainNoPain')
    pathTrainPain = '{}Training/Pain/'.format(pathData)
    x_TrainPain, y_TrainPain = directorySearch(pathTrainPain, 1, 'Train Pain', dataAugmentation=False)
    verifyLength(x_TrainPain, y_TrainPain, 'x_TrainPain', 'y_TrainPain')
    # rebalance classes for training data
#    print('Original Training pain shape\nx: {}\ny: {}'.format(np.asarray(x_TrainPain).shape, np.asarray(y_TrainPain).shape))
#    print('Original training no pain shape\nx: {}\ny: {}'.format(np.asarray(x_TrainNoPain).shape, np.asarray(y_TrainNoPain).shape))
#    x_TrainNoPain, y_TrainNoPain = x_TrainNoPain[0:-11376], y_TrainNoPain[0:-11376]
#    print('New training pain shape\nx: {}\ny: {}'.format(np.asarray(x_TrainNoPain).shape, np.asarray(y_TrainNoPain).shape))
    # find which class has more
    lenDiff = abs(len(x_TrainPain)-len(x_TrainNoPain))
    if len(x_TrainPain) < len(x_TrainNoPain):
        x_TrainNoPain, y_TrainNoPain = sklearn.utils.shuffle(x_TrainNoPain, y_TrainNoPain)
        x_TrainNoPain, y_TrainNoPain = x_TrainNoPain[0:-lenDiff], y_TrainNoPain[0:-lenDiff]
    else: 
        x_TrainPain, y_TrainPain = sklearn.utils.shuffle(x_TrainPain, y_TrainPain)
        x_TrainPain, y_TrainPain = x_TrainPain[0:-lenDiff], y_TrainPain[0:-lenDiff]

    # get val data
    pathValNoPain = '{}Validaiton/No_pain/'.format(pathData)
    x_ValNoPain, y_ValNoPain = directorySearch(pathValNoPain, 0, 'Val NoPain')
    verifyLength(x_ValNoPain, y_ValNoPain, 'x_ValNoPain', 'y_ValNoPain')
    pathValPain = '{}Validaiton/Pain/'.format(pathData)
    x_ValPain, y_ValPain = directorySearch(pathValPain, 1, 'Val Pain')
    verifyLength(x_ValPain, y_ValPain, 'x_ValPain', 'y_ValPain')

    # setup testing data
    test_x = np.asarray(x_TestNoPain + x_TestPain)
#    test_x_Landmarks = np.asarray(x_TestNoPain_Landmarks + x_TestPain_Landmarks)
    test_y = np.asarray(y_TestNoPain + y_TestPain)
#    test_x, test_y = sklearn.utils.shuffle(test_x, test_y)
    
    # setup training data
    train_x = np.asarray(x_TrainNoPain + x_TrainPain)
#    train_x_Landmarks = np.asarray(x_TrainNoPain_Landmarks + x_TrainPain_Landmarks)
    train_y = np.asarray(y_TrainNoPain + y_TrainPain)
#    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)

    # setup validation data
    val_x = np.asarray(x_ValNoPain + x_ValPain)
#    val_x_Landmarks = np.asarray(x_ValNoPain_Landmarks + x_ValPain_Landmarks)
    val_y = np.asarray(y_ValNoPain + y_ValPain)
#    val_x, val_y = sklearn.utils.shuffle(val_x, val_y)
    
    # normalize x data
    test_x = test_x.astype('float32')/255.0
    train_x = train_x.astype('float32')/255.0
    val_x = val_x.astype('float32')/255.0
    
    return test_x, test_y, train_x, train_y, val_x, val_y

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
        os.path.isfile(os.path.join(base, n))]

def buildModel(pathBase):
#     create model
    model = keras.models.Sequential()

#     2 layers of convolution
    model.add(keras.layers.Conv2D(64, 3, activation='relu', input_shape=(128,128,3)))
    model.add(keras.layers.BatchNormalization())
#     dropout
#    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    # dropout
#    model.add(keras.layers.Dropout(0.25))
    
#     max pooling
    model.add(keras.layers.MaxPooling2D())
    
#     2 layers of convolution
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
#     max pooling
    model.add(keras.layers.MaxPooling2D())
    
#     3 layers of convolution
    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())

#     max pooling
    model.add(keras.layers.MaxPooling2D())

    # ConvLSTM2D
#    model.add(keras.layers.ConvLSTM2D(64, 3, activation='relu'))
#     flatten
    model.add(keras.layers.Flatten())
##
##    # LSTM
##    model.add(LSTM(64, input_shape=(1016064,1), return_sequences=True))
#    
    # fully connected layer
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
#    model.add(keras.layers.Dense(1024, activation='relu'))
#    
    # dropout
    model.add(keras.layers.Dropout(0.99))

#    model.add(keras.layers.Dense(2, activation='relu'))
    
#    # final dense layer
    model.add(keras.layers.Dense(
#            1
            2
#                                 , activation='sigmoid' 
                                 , activation='softmax' 
#                                 , kernel_regularizer=regularizers.l2(0.01)
#                                 , activity_regularizer=regularizers.l1(0.01)
                                 ))    
    
    # resume from checkpoint
#    savedModelFiles = find_files(pathBase, '2019-02-07--*.hdf5')
#    if len(savedModelFiles) > 0:
#        if len(savedModelFiles) > 1:
#            print('Error: There are multiple saved model files.')
#            return
#        print("Resumed model's weights from {}".format(savedModelFiles[-1]))
#        # load weights
#        model.load_weights(os.path.join(pathBase, savedModelFiles[-1]))
#    model.summary()            
    # multiple GPUs
#    model = multi_gpu_model(model, gpus=16)
    
    # compile
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), 
#                  loss=keras.losses.binary_crossentropy, 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['acc']
#                  metrics=['acc', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity']
                  )
    
    return model

#    model = Sequential()
#    x_input = Input(shape=(128, 128, 3))
#    x_output = Conv2D(filters=64, kernel_size=3, activation='relu')(x_input)
#    base_model = Model(x_input, x_output)
#    model.add(TimeDistributed(base_model, input_shape=base_model.input_shape))
#    model.add(TimeDistributed(Flatten(input_shape=base_model.input_shape[1:])))
#    model.add(LSTM(2, activation='relu', recurrent_activation='hard_sigmoid', dropout=0.2))
#    model.add(LSTM(64, return_sequences=True))
#    model.add(Dense(2, activation='softmax'))
##    model = keras.applications.nasnet.NASNetLarge(weights = "imagenet", include_top=False, input_shape=(128, 128, 3))
#    model = keras.applications.Xception(weights = "imagenet", include_top=False, input_shape=(128, 128, 3))
#    model = keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape=(128, 128, 3))
#    print('number of layers: {}'.format(len(model.layers)))
#    model.summary()
#    for layer in model.layers[:36]:
#    for layer in model.layers:
#        layer.trainable=False
###    #Adding custom Layers 
    
#    input = Input((128,128,3))
#    x = Conv2D(filters=64, kernel_size=3, activation='relu')(input)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
#    x = BatchNormalization()(x)
#    x = MaxPooling2D()(x)
#    
#    x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
#    x = BatchNormalization()(x)
#    x = MaxPooling2D()(x)
#    
#    x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=256, kernel_size=3, activation='relu')(x)
#    x = BatchNormalization()(x)
#    x = MaxPooling2D()(x)
#    
#    x = Flatten()(x)
#    x = Dense(1024, activation="relu")(x)
#    x = Reshape((1, 127008//8))(x)
#    x = Reshape((1, 15876))(x)
#    x = LSTM(512)(x)
#    x = Dropout(0.9)(x)
#    output = Dense(2, activation='softmax')(x)
#    output = Dense(1, activation='sigmoid')(x)
#    model = Model(input, output)
#    model.summary()
#    x = model.output
#    x = Reshape((4, 4*2048))(x)
#    x = LSTM(1024)(x)
##    x = Conv2D(64, (3,3), activation='relu')(x)
##    x = BatchNormalization()(x)
##    x = Conv2D(64, (3,3), activation='relu')(x)
##    x = BatchNormalization()(x)
#    
#    x = Flatten()(x)
#    
#    input_lay = Input(shape=(None, 128, 128, 3)) #dimensions of your data
#    time_distribute = TimeDistributed(Lambda(lambda a: model(a)))(input_lay) # keras.layers.Lambda is essential to make our trick work :)
#    lstm_lay = LSTM(4)(time_distribute)
#    output_lay = Dense(2, activation='softmax')(lstm_lay)
    
#    x = LSTM(64)(x)
#    x = Dense(1024, activation="relu")(x)
#    x = Dropout(0.5)(x)
#    x = Dense(1024, activation="relu")(x)
#    x = Dropout(0.5)(x)
#    predictions = Dense(2, activation="softmax")(x)
#    # creating the final model 
#    model = Model(inputs = model.input, outputs = predictions)
#    model = Model(inputs=[input_lay], outputs=[output_lay])
    
#     multiple GPUs
    model = multi_gpu_model(model, gpus=16)
    # compile
    model.compile(
            loss = "sparse_categorical_crossentropy", 
#            loss = 'binary_crossentropy',
#            optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
#            optimizer = 'rmsprop', 
            optimizer=keras.optimizers.Adam(lr=0.0001),
            metrics=["accuracy"])
    
    return model



def RandomForest(model, train_x, train_x_Landmarks, train_y, test_x, test_x_Landmarks, test_y):

    #layer_name = 'dense_2'
    #extract = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    #features = extract.predict(train_x)

    clf = RandomForestClassifier(n_estimators=10).fit(train_x_Landmarks, train_y)
#    clf = SVC(kernel='rbf', C=10, verbose=False).fit(features, train_y)
#    predict_y = clf.predict(extract.predict(test_x))
    predict_y = clf.predict(extract.predict(test_x))

#     confusion matrix, classification accuracy, precision, recall, and binary F1 score
    conf_mat = confusion_matrix(test_y, predict_y)
    acc = sklearn.metrics.accuracy_score(test_y, predict_y)
    precision = sklearn.metrics.precision_score(test_y, predict_y)
    recall = sklearn.metrics.recall_score(test_y, predict_y)
    f1_score = sklearn.metrics.f1_score(test_y, predict_y)
    
    print('Confusion matrix (RF):\n{}\nClassification accuracy:{}\nPrecision:\t\t{}\nRecall:\t\t\t{}\nBinary F1 score: \t{}'.format(
            conf_mat, 
            acc,
            precision,
            recall,
            f1_score
            ))



    
def Exp3Fusion(model, train_x, train_x_Landmarks, train_y, test_x, test_x_Landmarks, test_y):
#    keras.backend.clear_session()
#    model.summary()
    layer_name = 'dense_2'
    extract = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = extract.predict(train_x)
#    extract = []
#    features = []
    
#    for inbound_node in range(0, len(model.get_layer(layer_name)._inbound_nodes)):
##        tempModel = Model(inputs=model.input, outputs=model.get_layer(layer_name).get_output_at(inbound_node))
###        tempModel = K.function([model.layers[0].input], [model.get_layer(layer_name).get_output_at(-1)])
##        tempModel2 = multi_gpu_model(tempModel, gpus=16)
###        extract.append([tempModel])
##        tempFeatures = tempModel2.predict(train_x)
###        tempFeatures = tempModel2([train_x])[0]
##        features.append([tempFeatures])
#        try:
#            tempModel = Model(inputs=model.input, outputs=model.get_layer(layer_name).get_output_at(inbound_node))
#            tempModel = multi_gpu_model(tempModel, gpus=16)
#            extract.append([tempModel])
#            tempFeatures = tempModel.predict(train_x)
#            features.append([tempFeatures])
#            print('Good node: {}'.format(inbound_node))
#        except:
#            print('Bad node: {}'.format(inbound_node))
#    inp = model.input                                           # input placeholder
#    functor = K.function([inp, K.learning_phase()], outputs=model.get_layer(layer_name).output )   # evaluation function
    
    # Testing
#    features = functor([train_x, 1.])
#    print(layer_outs)
        
#    clf = RandomForestClassifier(n_estimators=10).fit(features,train_y)
    training_data = np.concatenate((features, train_x_Landmarks), axis=1)
    clf = RandomForestClassifier(n_estimators=10).fit(training_data, train_y)
#    clf = SVC(kernel='rbf', C=10, verbose=False).fit(features, train_y)
#    predict_y = clf.predict(extract.predict(test_x))
    testing_data = np.concatenate((extract.predict(test_x), test_x_Landmarks), axis=1)
    predict_y = clf.predict(testing_data)

#     confusion matrix, classification accuracy, precision, recall, and binary F1 score
    conf_mat = confusion_matrix(test_y, predict_y)
    acc = sklearn.metrics.accuracy_score(test_y, predict_y)
    precision = sklearn.metrics.precision_score(test_y, predict_y)
    recall = sklearn.metrics.recall_score(test_y, predict_y)
    f1_score = sklearn.metrics.f1_score(test_y, predict_y)
    
    print('Confusion matrix (RF):\n{}\nClassification accuracy:{}\nPrecision:\t\t{}\nRecall:\t\t\t{}\nBinary F1 score: \t{}'.format(
            conf_mat, 
            acc,
            precision,
            recall,
            f1_score
            ))
    
if __name__ == "__main__":
#    pathBase = 'pain_classification/'
    pathBase = '/data/scanavan1/AffectiveComputing/Project2/pain_classification/'
    p_path, np_path = get_landmark_paths(LANDMARK_DATA)
    p_f, p_land = get_pain_landmarks(p[0:10])
    np_f, np_land = get_no_pain_landmarks(np[0:10])
#    print(np_f)
    ipp, inpp = get_imgs(IMG_DATA, p_f, np_f)
#    print('p type: {}, np type: {}\np_f type: {}, p_land type: {}\nnp_f type: {}, np_land type: {}'.format(type(p), type(np), type(p_f), type(p_land), type(np_f), type(np_land)))
    print('p shape: {}, np shape: {}\np_f shape: {}, p_land shape: {}\nnp_f shape: {}, np_land shape: {}'.format(p_path.shape, np_path.shape, p_f.shape, p_land.shape, np_f.shape, np_land.shape))
    sys.exit("done")
#    print('Image reading started at {}'.format(str(datetime.datetime.now())))
#    test_x, test_y, train_x, train_y, val_x, val_y = readImages(pathBase)
#    test_x, test_x_Landmarks, test_y, train_x, train_x_Landmarks, train_y, val_x, val_x_Landmarks, val_y = readImages(pathBase)
#    print('Image reading finished at {}'.format(str(datetime.datetime.now())))

#    print('Class balance started at {}'.format(str(datetime.datetime.now())))
#    unique, counts = np.unique(test_y, return_counts=True)
#    print('test_y: {}'.format(dict(zip(unique, counts))))
#    unique, counts = np.unique(train_y, return_counts=True)
#    print('train_y: {}'.format(dict(zip(unique, counts))))
#    unique, counts = np.unique(val_y, return_counts=True)
#    print('val_y: {}'.format(dict(zip(unique, counts))))
#    print('Class balance finished at {}'.format(str(datetime.datetime.now())))

#    print('Model building started at {}'.format(str(datetime.datetime.now())))
    keras.backend.clear_session()
    model = buildModel(pathBase)
#    print('Model building finished at {}'.format(str(datetime.datetime.now())))
    
#    print('Model evaluation started at {}'.format(str(datetime.datetime.now())))
    # fit model to data
    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
#    checkpoint = ModelCheckpoint('{0}{1}_{{epoch:02d}}-{{val_acc:.2f}}.hdf5'.format(pathBase, time),monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # checkpoint = ModelCheckpoint('model.hdf5'.format(pathBase, time),monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # earlyStop = EarlyStopping('val_acc',0.001,25)
    # callbacks_list = [checkpoint, earlyStop]
   # callbacks_list = [earlyStop]

    model.fit(x=train_x, y=train_y, batch_size=1, epochs=100, verbose=2, 
              # callbacks=callbacks_list,
              validation_data=(val_x, val_y),
              initial_epoch=0)
#    print(model.evaluate(test_x, test_y))
#    test_y_prob = model.predict(test_x)
#    test_y_pred = np.round(test_y_prob)
#    test_y_pred = np.argmax(test_y_prob, axis=-1)
#    print('Confusion matrix (CNN):\n{}'.format(confusion_matrix(test_y, test_y_pred)))
    RandomForest(model,train_x,train_y,test_x,test_y)
#    print('Model evaluation finished at {}'.format(str(datetime.datetime.now())))