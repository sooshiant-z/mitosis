## Author Sooshiant Zakariapour
## OpenCV and ELM library used 
## will add references and clean the code soon, sorry. Files refer to my local drive
## send me an email if I forgot

from elm import ELMClassifier
from elm import ELMRegressor
from sklearn import linear_model
import numpy as np, os
import mahotas.features
from skimage.feature import local_binary_pattern
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2

radius = 3
run = 0
n_points = 8 * radius


def print_Pr_Re_f1(EXP_NAME, GUESS, GROUND_TRUTH):
    TP = (GROUND_TRUTH + GUESS) == 2
    FP = (GROUND_TRUTH - GUESS) == -1
    TN = (GROUND_TRUTH + GUESS) == 0
    FN = (GROUND_TRUTH - GUESS) == 1
    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)
    TN = np.sum(TN)

    print EXP_NAME + ': Pr=' + str(float(TP) / (TP + FP)) + ' Re=' + str(float(TP) / (TP + FN)) + ' F1=' + str(2 * float(TP) / (2 * TP + FN + FP))
    return

def itemfreq(a):
    items, inv = np.unique(a, return_inverse=True)
    freq = np.bincount(inv)
    return np.array([items, freq]).T


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.001, 0.01):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
#X_Standardized = (Xlist - XMEAN) * (1.0 / XSTD)
#with open('xmeanxstd.pkl', 'rb') as f:
    #XMEAN, XSTD = cPickle.load(f)
m_shape1 = np.array(cv2.imread("F:/SHAPES/A03_01Cb_127.png"))
m_shape2 = np.array(cv2.imread("F:/SHAPES/A11_06Bb_408.png"))
m_shape3 = np.array(cv2.imread("F:/SHAPES/A03_05Da_232.png"))
m_shape4 = np.array(cv2.imread("F:/SHAPES/A05_01Ca_75.png"))
m_shape5 = np.array(cv2.imread("F:/SHAPES/A07_02Bc_1449.png"))
m_shape6 = np.array(cv2.imread("F:/SHAPES/A07_03Bb_683.png"))
m_shape7 = np.array(cv2.imread("F:/SHAPES/A11_05Ab_475.png"))
m_shape8 = np.array(cv2.imread("F:/SHAPES/A04_00Db_142.png"))

def Calc_feats(img):
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    # cv2.imshow("Original", img[:,:,0])
    # cv2.imshow("LBP", lbp / 25)
    # x = m_shape[:,:,0]
    templ1 = cv2.matchTemplate(img, m_shape1[:, :, 2], cv2.TM_CCOEFF)
    templ2 = cv2.matchTemplate(img, m_shape2[:, :, 2], cv2.TM_CCOEFF)
    templ3 = cv2.matchTemplate(img, m_shape3[:, :, 2], cv2.TM_CCOEFF)
    templ4 = cv2.matchTemplate(img, m_shape4[:, :, 2], cv2.TM_CCOEFF)
    templ5 = cv2.matchTemplate(img, m_shape5[:, :, 2], cv2.TM_CCOEFF)
    templ6 = cv2.matchTemplate(img, m_shape6[:, :, 2], cv2.TM_CCOEFF)
    templ7 = cv2.matchTemplate(img, m_shape7[:, :, 2], cv2.TM_CCOEFF)
    templ8 = cv2.matchTemplate(img, m_shape8[:, :, 2], cv2.TM_CCOEFF)

    # Gabor_feats = compute_feats(img[:,:,0 ], kernels)
    # cv2.imshow("X", result)
    # x = itemfreq(lbp.ravel())
    x = np.histogram(np.asarray(img, dtype=np.int), bins=xrange(0, 27),  density=False)[0]


    # hist = x[:, 1]/sum(x[:, 1])

    # lbp2 = lbp[16:48,16:48] #local_binary_pattern(img[16:48,16:48,0], n_points, radius, method='uniform')
    # x2 = itemfreq(lbp2.ravel())
    # x3 = np.histogram(lbp2.ravel() , bins=xrange(0,26))
    # hist2 = x2[:, 1]/sum(x2[:, 1])
    x2 = np.histogram(img[16:48, 16:48], bins=xrange(0, 27), density=False)[0]

    # cv2.imshow("image",lbp2)
    new_haralick = mahotas.features.haralick(img)
    # haralick_feature = mahotas.features.haralick(img[:,:,0]).mean(0)
    # Xlist.append(np.append(np.append(haralick_feature ,blue_histogram),  lbp.flatten()))
    # Xlist.append(np.append(np.append(x2,haralick_feature),x))
    Entropy = np.zeros((1))
    #cv2.imshow('orig', img[:,:,2])
    #cv2.imshow('ENT', entropy(img[:,:,2], disk(4)))
    Entropy[0] = (entropy(img, disk(4)).sum().astype(np.float))

    return np.concatenate((Entropy, x2, new_haralick[0, :], new_haralick[1, :], new_haralick[2, :],
                                 new_haralick[3, :], x,
                                 templ1[0], templ2[0], templ3[0], templ4[0], templ6[0], templ7[0], templ8[0],
                                 templ5[0],
                                 # Gabor_feats[:,0] , Gabor_feats[:,1],
                                 ), axis=0)



if __name__ == '__main__':



    path = "F:/TRAINING_BALANCED - MODIFIED/"

    dir = os.listdir(path)
    class1 = os.listdir(path + dir[0])
    np.random.shuffle(class1)
    class1 = class1[:8000]
    class2 = os.listdir(path + dir[1])
    np.random.shuffle(class2)
    class2 = class2[:class1.__len__()]
    T_1 = np.ones(class1.__len__())
    T_2 = np.zeros(class2.__len__())
    master = np.hstack((class1, class2))
    T_outputs = np.hstack((T_1, T_2))


    SHUFFLE_ME = np.stack((master, T_outputs), axis=1)
    np.random.shuffle(SHUFFLE_ME)

    X_names = SHUFFLE_ME[:, 0]
    T_outputs = np.asarray(SHUFFLE_ME[:, 1], dtype=np.float)
    ###T_outputs = np.stack((T_outputs, (T_outputs * -1 + 1)), axis=1)
    total_items_count = X_names.__len__()

    X = np.zeros((total_items_count/2, 4096))
    X_neg = np.zeros((total_items_count/2, 4096))
    T = np.zeros((total_items_count, 2))  #


    names_train_x = X_names[:total_items_count*9/10]
    train_y = T_outputs[:total_items_count*9/10]
    names_test_x  = X_names[total_items_count*9/10:total_items_count]
    test_y = T_outputs[total_items_count*9/10:total_items_count]
    #names_val_x = X_names[total_items_count*9/10:total_items_count]
    #val_y = T_outputs[total_items_count*9/10:total_items_count]
    train_x = np.zeros((names_train_x.__len__(),64*64),dtype=np.float)
    test_x = np.zeros((names_test_x.__len__(),64*64),dtype=np.float)


    FEAT_train_x = np.zeros((names_train_x.__len__(),113),dtype=np.float)
    FEAT_test_x = np.zeros((names_test_x.__len__(),113),dtype=np.float)
    #val_x = np.zeros((names_val_x.__len__(),64*64),dtype=np.float)

    for i in xrange(0,train_x.__len__()):
        if names_train_x[i][0] == 'M':
            INP_IMG = cv2.imread(path + 'MITOSIS/' + names_train_x[i])[:,:,2]
            train_x[i] = INP_IMG.flatten()
            FEAT_train_x[i] = Calc_feats(INP_IMG)
        else:
            INP_IMG = cv2.imread(path + 'NON_MITOSIS/' + names_train_x[i])[:,:,2]
            train_x[i] = INP_IMG.flatten()
            FEAT_train_x[i] = Calc_feats(INP_IMG)

    for i in xrange(0,test_x.__len__()):
        if names_test_x[i][0] == 'M':
            INP_IMG = cv2.imread(path + 'MITOSIS/' + names_test_x[i])[:,:,2]
            test_x[i] = INP_IMG.flatten()
            FEAT_test_x[i] = Calc_feats(INP_IMG)
        else:
            INP_IMG = cv2.imread(path + 'NON_MITOSIS/' + names_test_x[i])[:,:,2]
            test_x[i] = INP_IMG.flatten()
            FEAT_test_x[i] = Calc_feats(INP_IMG)

#    for i in xrange(0,val_x.__len__()):
#        if names_val_x[i][0] == 'M':
#            val_x[i] = cv2.imread(path + 'MITOSIS/' + names_val_x[i])[:,:,2].flatten()
#        else:
#            val_x[i] = cv2.imread(path + 'NON_MITOSIS/' + names_val_x[i])[:,:,2].flatten()

    # Load data sets
    #train_x, train_y, val_x, val_y, test_x, test_y = get_datasets(load_mnist())
    # Build ELM

    FEAT_cls = ELMClassifier(n_hidden=900,
                        alpha=0.93,
                        activation_func='multiquadric',
                        regressor=linear_model.Ridge(),
                        random_state=21398023)

    FEAT_cls.fit(FEAT_train_x, train_y)

    REGRSR = ELMRegressor(n_hidden=900,
                        alpha=0.93,
                        activation_func='multiquadric',
                        regressor=linear_model.Ridge(),
                        random_state=21398023)

    REGRSR.fit(train_x, train_y)

#    val_x = val_x / 255.
    cls = ELMClassifier(n_hidden=900,
                        alpha=0.93,
                        activation_func='multiquadric',
                        regressor=linear_model.Ridge(),
                        random_state=21398023)
    cls.fit(train_x, train_y)

    RGRS_feat = ELMRegressor(n_hidden=900,
                        alpha=0.93,
                        activation_func='multiquadric',
                        regressor=linear_model.Ridge(),
                        random_state=21398023)
    RGRS_feat.fit(FEAT_train_x, train_y)
    # Evaluate model

    print_Pr_Re_f1('Feature-less CLS accuracy:', cls.predict(test_x), test_y)
    print_Pr_Re_f1('Feature-based CLS accuracy:', FEAT_cls.predict(FEAT_test_x), test_y)

    DUAL_CLS = ELMClassifier(n_hidden=900,
                        alpha=0.93,
                        activation_func='multiquadric',
                        regressor=linear_model.Ridge(),
                        random_state=21398023)
    DUAL_CLS.fit(np.column_stack((REGRSR.predict(train_x), RGRS_feat.predict(FEAT_train_x))), train_y)

    print_Pr_Re_f1('DUAL_CLS accuracy:', DUAL_CLS.predict(np.column_stack((REGRSR.predict(test_x), RGRS_feat.predict(FEAT_test_x)))), test_y)


    path_06 = "F:/A06T/test/0/"
    indx = 0
    test = np.zeros((2200,4096), dtype=np.float)
    FT_test = np.zeros((2200,113),dtype=np.float)

    for file in os.listdir(path_06):
        INP_IMG = cv2.imread(path_06 + file )[:,:,2]
        test[indx] = INP_IMG.flatten()
        FT_test[indx] = Calc_feats(INP_IMG)
        indx += 1

    j = DUAL_CLS.predict(np.column_stack((REGRSR.predict(test[:indx]), RGRS_feat.predict(FT_test[:indx]))))
    indx = 0
    for file in os.listdir(path_06):
        if j[indx] == 1.0:
            print 'copy ' + file + ' ..\\DUAL0\\'
        indx += 1

#    print 'Validation accuracy:', cls.score(val_x, val_y)
    print "Fin."
