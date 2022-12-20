#!/usr/bin/env python
# coding: utf-8

##importing packages
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


## create a preprocessor
preprocessor = create_preprocessor('xception', target_size=(150, 150))


## initialize
interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

## post processing
classes = [
    'cups',
    'glass',
    'plate',
    'spoon',
    'knife',
    'fork'
]


## download the image
#url = 'https://github.com/bhasarma/kitchenware-classification-project/blob/main/test-image.jpg?raw=true'
# function that will get and download image from url
def predict(url):
    X = preprocessor.from_url(url)

    ## doing prediction
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result