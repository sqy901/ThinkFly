import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import glob
import os


class DIGITS_CLASSIFIER_LITE():
    def __init__(self, model_path):
        interpreter = tflite.Interpreter(model_path)
        interpreter.allocate_tensors()
        self.digits_classifier = interpreter.get_signature_runner('serving_default')

    def preprocess_image(self, images):
        gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        resized_images = np.array([cv2.resize(gray_image, (32, 32)) for gray_image in gray_images])
        normalized_images = resized_images.astype('float32') / 255.0
        return np.expand_dims(normalized_images, axis=-1)

    def postprocess_predictions(self, predictions):
        predictions = predictions['tf.stack']
        predicted_labels = [''.join(map(str, row)) for row in np.argmax(predictions, axis=2)]
        predicted_confs = np.min(np.max(predictions, axis=2), axis=1)
        return predicted_labels, predicted_confs

    def predict(self, images):
        predictions = self.digits_classifier(input_1=self.preprocess_image(images))
        return self.postprocess_predictions(predictions)

    def predict_without_preprocess(self, preprocessed_images):
        predictions = self.digits_classifier(input_1=preprocessed_images)
        return self.postprocess_predictions(predictions)

    def _read_images_from_directory(self, directory):
        pattern = os.path.join(directory, "*.*")
        png_files = glob.glob(pattern)
        return [cv2.imread(file) for file in png_files]

    def predict_from_directory(self, directory):
        images = self._read_images_from_directory(directory)
        return self.predict(images)


# example
if __name__ == '__main__':
    digits_classifier_lite = DIGITS_CLASSIFIER_LITE('models/nb_model.tflite')

    # random_array = np.random.rand(1000,32,32,1).astype(np.float32)
    # predicted_labels, predicted_confs = digits_classifier_lite.predict_without_preprocess(random_array)
    # print(predicted_labels, predicted_confs)

    num = [0] * 100

    predicted_labels, predicted_confs = digits_classifier_lite.predict_from_directory('generate_image')
    for i in range(len(predicted_labels)):
        if predicted_confs[i] > 0.5:
            num[int(predicted_labels[i])] += 1
    for i in range(1, 100):
        if num[i] > 0:
            print("{}:{}".format(i, num[i]))