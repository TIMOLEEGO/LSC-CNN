from model import ADresnetBuilder
import cv2
import numpy as np
import time
import math
import tensorflow as tf

builder = ADresnetBuilder()
model = builder.build_resnet18()
input_shape = (None, None, 1)
model.build(input_shape)
model.load_weights('model_37_4.h5')
cap = cv2.VideoCapture('real.avi')  #change test file
i = 0
ret = 1
total_psnr = 0
total_ssim = 0
total_time = 0
while (ret):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        G_col = np.random.normal(0, 15 / 255., (1, 320))
        noise = np.expand_dims(np.tile(G_col, (240, 1)), axis=2)
        img = np.expand_dims(frame,axis=2).astype(np.float)/255.
        noise_img = np.expand_dims((img + noise),axis=0)
        #cv2.imshow('1',(img + noise))
        cv2.imwrite('./test/noise' + '/' + '%d.png' % i, (img + noise))
        starttime = time.clock()
        result = model.predict(noise_img)
        single_time = time.clock() - starttime
        print(single_time)
        #-------------------------------PSNR------------------------------------
        raw_img_data = np.array(img,dtype=np.float64)
        denoise_img_data = np.array(result,dtype=np.float64)
        #----------------------------MSE----------------------------------------
        mse = (np.abs(raw_img_data-denoise_img_data)**2.).mean()
        psnr_denoise_raw = 20*np.log10(1./np.sqrt(mse))
        print(psnr_denoise_raw)
        # -------------------------SSIM------------------------------------------
        im1 = tf.image.convert_image_dtype(raw_img_data, tf.float32)
        im2 = tf.image.convert_image_dtype(denoise_img_data, tf.float32)
        ssim_tf = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        ssim_np = ssim_tf.numpy()
        ssim = ssim_np[0]
        print("ssim:",ssim)
        total_psnr += psnr_denoise_raw
        total_ssim += ssim
        #-----------------------------------------------------------------------
        result = np.clip(np.squeeze(result)*255,0,255).astype(np.uint8)
        total_time += single_time
        cv2.imwrite('./test/real' + '/' + '%d.png' % i, result)
        i += 1
        cv2.waitKey(1)
print("avg_psnr:", total_psnr/269) #real.avi->269frame
print("avg_ssim:", total_ssim/269)
print("avg_time:", total_time/269)