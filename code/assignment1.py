import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randrange


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Canny Edge dedection
def Canny_edge(img):
    # Canny Edge
    canny_edges = cv2.Canny(img, 70, 250, L2gradient=True)
    return canny_edges


# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)

    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    # As = [0, 45, 90, 135]
    As = [0, 30, 60, 90, 120, 150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1.5, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)
    # plt.show()
    return out


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def accuracy_calculator(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if (actual[i] == predicted[i]):
            correct += 1
    return correct / (float(len(actual))) * 100


"""
def euclideanDistance_OLD(p1,p2):  #for ex-> p1=dataset[0] vs p2=dataset[1]
    distance=np.sum(np.square(p1[0] - p2[0]))
    return np.sqrt(distance)
"""


def euclideanDistance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def k_fold_cross_validation_split(dataset, k):
    splitted_dataset = list()
    temp_dataset = list(dataset)
    fold_size = len(dataset) // k
    index = 0
    for i in range(k):
        fold = list()
        while (len(fold) < fold_size):  # put the np array and its target to the fold
            index = randrange(len(temp_dataset))
            fold.append(temp_dataset.pop(index))
        splitted_dataset.append(fold)
    return splitted_dataset


def get_nearest_neighbors(dataset, test_set, k_neighbor):
    distances = list()
    for index in range(len(dataset)):
        distance = euclideanDistance(test_set[0], dataset[index][0])
        distances.append([dataset[index], distance])
    distances.sort(key=lambda x: x[1])  # sort by distance
    neighbors = distances[:k_neighbor]
    return neighbors  # we start from second index
    # because first is itself


def predict(dataset, test, k_neighbor):
    neighbors = get_nearest_neighbors(dataset, test, k_neighbor)
    target_vals = [row[0][-1] for row in neighbors]  # prediction of nearest

    # knn prediction
    prediction = max(set(target_vals), key=target_vals.count)

    # from here to end we calculate the weighted knn prediction
    f1 = 0  # covid
    f2 = 0  # normal
    f3 = 0  # viral
    for n in neighbors:  # calculate which is higher frequency
        if (n[1] == 0):	#if there is an identical value for extreme conditions
            return prediction, n[0][-1]
        if n[0][-1] == 1:
            f1 += (1 / n[1])
        elif n[0][-1] == 2:
            f2 += (1 / n[1])
        elif n[0][-1] == 3:
            f3 += (1 / n[1])
    w_p = f1
    weighted_prediction = 1
    if (f1 < f2):
        w_p = f2
        weighted_prediction = 2
    if (w_p < f3):
        w_p = f3
        weighted_prediction = 3

    return prediction, weighted_prediction


def validation_predict(dataset, test_set, k_neighbor):  # this is written for validation accuracy
    predictions = list()  # there are k fold times prediction values
    weighted_predictions = list()
    for test in test_set:  # test it for each test sample
        prediction, weighted_prediction = predict(dataset, test, k_neighbor)
        predictions.append(prediction)
        weighted_predictions.append(weighted_prediction)
    return predictions, weighted_predictions  # return set of the knn and wknn predictions


def knn_algorithm(dataset, k_fold, k_neighbor):
    folds = k_fold_cross_validation_split(dataset, k_fold)
    scores = list()
    weighted_scores = list()
    index = 0 #to remove the one fold to test
    for fold in folds:
        # fold is 2d array with dataset and target set
        train_set = list(folds)
        train_set.pop(index)
        train_set = sum(train_set, [])  # to obtain one train set
        test_set = list()
        for row in fold:  # fold[0]=dataset     fold[1]=target_set
            copy = list(row)
            # test_set.clear()
            test_set.append(copy)
            copy[-1] = None  # to predict it
        predicted, weighted_predicted = validation_predict(train_set, test_set, k_neighbor)
        actual = [target_va[-1] for target_val in fold]  # validation_set targets
        accuracy = accuracy_calculator(actual, predicted)
        weighted_accuracy = accuracy_calculator(actual, weighted_predicted)
        scores.append(accuracy)
        weighted_scores.append(weighted_accuracy)
        index += 1
    return scores, weighted_scores


##Read training data


path0 = "C:\\Users\\xx\\Untitled Folder\\train";
dataset = []  # dataset and target value
Ycounter = 0
# Covid=1
# Normal=2
# Viral=3

# Read image  
for path in os.listdir(path0):
    folder = path0 + "\\" + path
    Ycounter += 1
    for filename in os.listdir(folder):
        img = cv2.imread(folder + "\\" + filename).astype(np.uint8)  # float32
        out = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        out = Gabor_process(out)
        # out=Canny_edge(out)
        # out=auto_canny(out)
        dataset.append([out, Ycounter])  # dataset[i][0=dataset
        # dataset[i][1]=target

    ##Validation accuracy calculations

k_folds = [2, 4, 5, 6, 8, 10, 16, 32]  # 1 is not needed
k_neighbors = [3, 4, 5, 6, 7, 8, 9]  # 1 is not accurate and 2 can be ambigious
accuracy_score = 0

for j in k_neighbors:
    scores, weighted_scores = knn_algorithm(dataset, 10, j)
    aa = sum(scores) / (float)(len(scores))
    if (aa > accuracy_score):
        accuracy_score = aa
        print(j)
        print("Accuracy: %.2f%% " % (accuracy_score))
    bb = sum(weighted_scores) / (float)(len(weighted_scores))
    if (bb > accuracy_score):
        accuracy_score = bb
        print(j)
        print("Weighted Accuracy: %.2f%% " % (accuracy_score))

    ##Calculate the test data
testpath="C:\\Users\\xx\\Untitled Folder\\test"
Ycounter=0
knn_counter=0
wknn_counter=0
test_counter=0
for path in os.listdir(testpath):
    foldername=testpath+"\\"+path
    Ycounter+=1
    for filename in os.listdir(foldername):
        img = cv2.imread(foldername+"\\"+filename).astype(np.uint16)#float32
        out=cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA )
        out=Gabor_process(out)
        x=[out,Ycounter]
        label_knn,label_wknn=predict(dataset,x,3)
        if (label_knn ==Ycounter):
            knn_counter+=1
        if (label_wknn==Ycounter):
            wknn_counter+=1
        test_counter+=1
print(knn_counter,wknn_counter,test_counter)
print("knn accuracy of test samples: %.3f%%" % (100*knn_counter /test_counter))
print("wknn accuracy of test samples: %.3f%%" % (100*wknn_counter /test_counter))
