import numpy as np
import matplotlib.pyplot as plt
from numpy.core.arrayprint import printoptions

from skimage.color import gray2rgb
from skimage import feature, transform

import sys
sys.path.append('/home/heedong/Documents/ABELE/')
import os
os.chdir("../")
print(os.getcwd())
import pickle

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder

from keras.datasets import mnist


import warnings
warnings.filterwarnings('ignore')


# def main():


random_state = 0
dataset = 'mnist'
black_box = 'RF'

ae_name = 'aae'

path = './'
path_models = path + 'models/'
path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

black_box_filename = path_models + '%s_%s' % (dataset, black_box)

_, _, X_test, Y_test, use_rgb = get_dataset(dataset)
bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)
ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
ae.load_model()

class_name = 'class'
class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]
cnt = 0
for i2e in range(2000):
# i2e = 0
    img = X_test[i2e]

    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type='rnd', use_prob=True, size=1000, ocr=0.1,
                    kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                    filter_crules=True, random_state=random_state, verbose=True, alpha1=0.5, alpha2=0.5,
                    metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                    bb_predict_proba=bb_predict_proba)

    exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)

    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    print(exp.bb_pred, exp.dt_pred, exp.fidelity)
    print(exp.limg)

    img2show, mask = exp.get_image_rule(features=None, samples=10)
    if use_rgb:
        plt.imshow(img2show, cmap='gray')
    else:
        plt.imshow(img2show)
        bbo = bb_predict(np.array([img2show]))[0]
        plt.title('image to explain - black box %s' % bbo)
        # plt.show()

    # if use_rgb:
    #     plt.imshow(img2show, cmap='gray')
    # else:
    #     plt.imshow(img2show)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, img2show.shape[1], dx)
    yy = np.arange(0.0, img2show.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)

    # Compute edges (to overlay to heatmaps later)
    percentile = 100
    dilation = 3.0
    alpha = 0.8
    xi_greyscale = img2show if len(img2show.shape) == 2 else np.mean(img2show, axis=-1)
    in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < 0.5] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan
    overlay = edges

    # # abs_max = np.percentile(np.abs(data), percentile)
    # # abs_min = abs_max

    # # plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    # plt.imshow(mask, extent=extent, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    # plt.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    # plt.axis('off')
    # plt.title('attention area respecting latent rule')
    # plt.show()

    # latent_dim = 4

    # plt.figure(figsize=(12, 4))
    # for i in range(latent_dim):
    #     img2show, mask = exp.get_image_rule(features=[i], samples=10)
    #     plt.subplot(1, 4, i+1)
    #     if use_rgb:
    #         plt.imshow(img2show)
    #     else:
    #         plt.imshow(img2show, cmap='gray')
    #     plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    #     plt.title('varying dim %d' % i)
    # plt.suptitle('attention area respecting latent rule')
    # plt.show()

    # prototypes = exp.get_prototypes_respecting_rule(num_prototypes=5, eps=255*0.25)
    # for pimg in prototypes:
    #     bbo = bb_predict(np.array([gray2rgb(pimg)]))[0]
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.title('prototype %s' % bbo)
    #     plt.show()

    # prototypes, diff_list = exp.get_prototypes_respecting_rule(num_prototypes=5, return_diff=True)
    # for pimg, diff in zip(prototypes, diff_list):
    #     bbo = bb_predict(np.array([gray2rgb(pimg)]))[0]
    #     plt.subplot(1, 2, 1)
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.title('prototype %s' % bbo)
    #     plt.subplot(1, 2, 2)
    #     plt.title('differences')
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.pcolormesh(range(diff.shape[0]), range(diff.shape[1]), diff, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    #     plt.show()

    cprototypes = exp.get_counterfactual_prototypes(eps=0.01)
    for cpimg in cprototypes:
        bboc = bb_predict(np.array([cpimg]))[0]
        if use_rgb:
            plt.imshow(cpimg)
        else:
            plt.imshow(cpimg, cmap='gray')
        plt.title('cf - black box %s' % bboc)
        # plt.show()

    cprototypes_interp = exp.get_counterfactual_prototypes(eps=0.01, interp=5)
    for cpimg_interp in cprototypes_interp:
        for i, cpimg in enumerate(cpimg_interp):
            bboc = bb_predict(np.array([cpimg]))[0]
            plt.subplot(1, 5, i+1)
            if use_rgb:
                plt.imshow(cpimg)
            else:
                plt.imshow(cpimg, cmap='gray')
            plt.title('%s' % bboc)
        fo = bb_predict(np.array([cpimg_interp[0]]))[0]
        to = bb_predict(np.array([cpimg_interp[-1]]))[0]
        
        from_save_path = './results/MNIST_from_image/'
        to_save_path = './results/MNIST_cf_image/'
        if not os.path.exists(from_save_path):
                os.mkdir(from_save_path)
        if not os.path.exists(from_save_path):
                os.mkdir(to_save_path)
        plt.imsave(from_save_path+str(cnt)+ '.jpeg', cpimg_interp[0])
        plt.imsave(to_save_path+str(cnt)+'_'+str(to)+'.jpeg', np.array(cpimg_interp[-1], dtype = 'uint8'))
        print("number", i2e, "finished")
        print("cnt", cnt, "imaged saved")
        cnt += 1


        # plt.suptitle('black box - from %s to %s' % (fo, to))
        # plt.show()
        
# bb_predict(np.array([cpimg_interp[0]]))[0]
# bb_predict(np.array([cpimg_interp[-1]]))[0]
# test = plt.imread('./results/MNIST_from_image/0.jpeg')
# bb_predict(np.array([test]))
# test_pre = plt.imread('./results/MNIST_cf_image/4_4.jpeg')
# plt.imshow(test_pre)
# bb_predict(np.array([test_pre]))
path_from_image = './results/MNIST_from_image/'
path_cf_image = './results/MNIST_cf_image/'
from_file_list = os.listdir(path_from_image)
cf_file_list = os.listdir(path_cf_image)
from_file_list.sort()
cf_file_list.sort()

for cf in cf_file_list:
    if int(cf.split('_')[1].split('.')[0]) == 3:
        original_read_path = path_from_image+cf.split('_')[0]+'.jpeg'
        origunal_read = plt.imread(original_read_path)
        print(cf.split('_')[0], cf)
        if bb_predict(np.array([original_read]))[0] == 7:
            print(original_read_path)

for cf in cf_file_list:
    cf_read = plt.imread(path_cf_image+cf)

    if bb_predict(np.array([cf_read]))[0] == 3:
        original_read_path = path_from_image+cf.split('_')[0]+'.jpeg'
        original_read = plt.imread(original_read_path)
        print('cf img : ',cf, 'finding img : ', cf.split('_')[0]+'.jpeg')
        if bb_predict(np.array([original_read]))[0] == 7:
            print('matched img : ',cf, original_read_path)
            plt.imshow(original_read)
            plt.show()
            plt.imshow(cf_read)
            plt.show()


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_test = np.stack([gray2rgb(x) for x in X_test.reshape((-1, 28, 28))], 0)
label_7_image = X_test[np.where(Y_test==7)]
for i, save_img in enumerate(label_7_image):
    plt.imsave('./results/test_label_7/'+str(i)+'.jpeg',save_img, cmap='gray')


# if __name__ == "__main__":
#     main()
