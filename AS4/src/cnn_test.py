import tensorflow as tf
import numpy as np
import cv2 as cv
import cnn


model_fn = cnn.cnn_model_fn

# Initializing model training variables
imgW, imgH, channels = 28, 28, 1
train_epochs = 5
batch_size = 200
drop_rate = 0.4
learn_rate = 0.001
cnn_layers = 2
model_dir = '../models/cnn_model'

save_summary_frequency = 1


# Different types of popular weight initialization methods
initializers = {
    'fast_conv': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True),
    'he_rec': tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
    'xavier': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
}
parameters = {'img_size': [imgH, imgW, channels], 'summary_steps': save_summary_frequency,
              'model_dir': model_dir, 'learn_rate': learn_rate, 'drop_rate': drop_rate,
              'w_inits': initializers, 'depth': cnn_layers}


# Create the Estimator
classifier = tf.estimator.Estimator(
    model_fn=tf.contrib.estimator.replicate_model_fn(model_fn), # Distributed Training on All GPUs:
    model_dir=model_dir,
    params=parameters
)

while True:
        userInput = input("Enter filename or q to quit")
        if userInput == "q":
            break
        else:
            filename = userInput
            # prediction data
            inputImage = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            cv.imshow("OriginalImage",inputImage)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            inputImage = cv.resize(inputImage,(28,28))

            blur = cv.GaussianBlur(inputImage,(5,5),0)
            thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,2)

            cv.imshow("Binarary Image",thresh)
            cv.waitKey(0)
            cv.destroyAllWindows()

            flat = thresh.flatten().reshape(1, 784) / 255.0
            npflat = np.zeros((1, 784))
            npflat = flat
            output = npflat.astype(np.float32)
            

            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": output},
                y=None,
                batch_size=1,
                num_epochs=1,
                shuffle=False,
                num_threads=1,
            )

            predict_results = classifier.predict(predict_input_fn)

            # print(predict_data)
            for idx, prediction in enumerate(predict_results):
                #print(prediction["classes"])
                if prediction["classes"] == 0:
                    print("Even\n")
                else:
                    print("Odd\n")

