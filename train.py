from model import ADresnetBuilder
import h5py
import numpy as np
from tensorflow.keras import callbacks
import random
import tensorflow.keras.optimizers as opt
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import datetime
import time
import cv2
import keras.backend as K
import tensorflow as tf

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.1#0.5
   epochs_drop = 40 #5.0
   lr = initial_lrate * (drop ** np.floor((1 + epoch) / epochs_drop))
   return lr
lr = LearningRateScheduler(step_decay) 

def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*128)
    return res

def data_generator(data_path,batch_size):
    h5f = h5py.File(data_path, 'r')
    keys = list(h5f.keys())
    flag=0
    batch_img = []
    batch_noiseimg = []
    random.shuffle(keys)
    for k in keys:
        img = np.array(h5f[k]).transpose((1,2,0))
        G_col = np.random.normal(0, 15 / 255., (1,50))
        noise = np.expand_dims(np.tile(G_col, (50, 1)),axis=2)
        noise_img = img+noise
        batch_img.append(img)
        batch_noiseimg.append(noise_img)
        flag+=1
        if flag==batch_size:
            yield (np.array(batch_noiseimg) ,np.array(batch_img))
            flag=0
            batch_img.clear()
            batch_noiseimg.clear()


log_dir = '../log/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
check_point = callbacks.ModelCheckpoint('../model.h5',
                                     monitor='loss',verbose=1,save_best_only=True, save_weights_only=False, mode='min')
vis = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_grads=True, write_images=True,
                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                            embeddings_data=None, update_freq='epoch')
builder = ADresnetBuilder()
model = builder.build_resnet18()
model.compile(optimizer=opt.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1.e-8), loss=custom_loss)

batch_size=128
h5f = h5py.File('../train.h5', 'r')
keys = list(h5f.keys())
steps = int(np.floor(len(keys) / batch_size))

epochs=50
summary_write = tf.summary.create_file_writer(log_dir)
for epoch in range(epochs):
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, mode='min', verbose=1)
    train_data=data_generator('../train.h5',batch_size=128)
    model.fit(train_data,steps_per_epoch=steps,use_multiprocessing=False,shuffle=True,callbacks=[check_point,vis,reduce_lr,lr])
    cap = cv2.VideoCapture('../dym4.avi')
    ret = 1
    total_psnr = 0  # 0710+
    total_mse = 0  # 0710+
    total_ssim = 0
    starttime = time.clock()
    while (ret):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            G_col = np.random.normal(0, 15 / 255., (1, 320))
            noise = np.expand_dims(np.tile(G_col, (240, 1)), axis=2)
            img = np.expand_dims(frame, axis=2).astype(np.float) / 255.
            noise_img = np.expand_dims((img + noise), axis=0)
            #cv2.imshow('1', (img + noise))

            # -------------------------------PSNR-MSE-----------------------------------
            result = model.predict(noise_img)
            raw_img_data = np.array(img, dtype=np.float32)  # dtype=np.float64)
            denoise_img_data = np.array(result, dtype=np.float32)  # dtype=np.float64)
            mse = (np.abs(raw_img_data - denoise_img_data) ** 2.).mean()
            psnr_denoise_raw = 20 * np.log10(1. / np.sqrt(mse))

            # -------------------------------SSIM-------------------------------------
            im1 = tf.image.convert_image_dtype(raw_img_data, tf.float32)
            im2 = tf.image.convert_image_dtype(denoise_img_data, tf.float32)

            ssim_tf = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
            ssim_np = ssim_tf.numpy()
            ssim = ssim_np[0]
            print("PSNR: ", psnr_denoise_raw, "    SSIM: ", ssim)
            with summary_write.as_default():
                tf.summary.scalar('PSNR step=epoch', psnr_denoise_raw, step=epoch)
                tf.summary.scalar('SSIM step=epoch', ssim, step=epoch)

            total_mse += mse
            total_psnr += psnr_denoise_raw
            total_ssim += ssim

    endtime = time.clock()
    totaltime = endtime - starttime
    AVG_TIME = totaltime / 269
    AVG_PSNR = total_psnr / 269
    AVG_MSE = total_mse / 269
    AVG_SSIM = total_ssim / 269
    
    print("AVG_PSNR: ", AVG_PSNR)
    print("AVG_MSE: ", AVG_MSE)
    print("AVG_TIME: ", AVG_TIME)
    print("AVG_SSIM: ", AVG_SSIM)

    with summary_write.as_default():
        tf.summary.scalar('AVG_PSNR step=epoch', AVG_PSNR, step=epoch)
        tf.summary.scalar('AVG_MSE step=epoch', AVG_MSE, step=epoch)
        tf.summary.scalar('AVG_SSIM step=epoch', AVG_SSIM, step=epoch)
        tf.summary.scalar('AVG_TIME step=epoch', AVG_TIME, step=epoch)

    if AVG_PSNR>=37:
       model.save('../model_37.h5')
    if AVG_PSNR>=37.2:
       model.save('../model_37_2.h5')
    if AVG_PSNR>=37.4:
       model.save('../model_37_4.h5')


