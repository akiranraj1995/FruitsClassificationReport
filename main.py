import os
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from PIL import Image as PILImage
import numpy as np
import tensorflow.lite as tflite
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.utils import platform
from plyer import filechooser

kivy.require("2.0.0")

class ImageClassifierApp(App):
    def build(self):
        self.title = "Fruits Image Classification Report"
        self.model = tflite.Interpreter(model_path="GaiApp_Epoch10.tflite")
        self.model.allocate_tensors()
        self.labels = self.load_labels("labels.txt")

        layout = BoxLayout(orientation="vertical", spacing=10)

        self.folder_input = TextInput(hint_text="Enter folder path or tap 'Select Folder'")
        layout.add_widget(self.folder_input)

        select_folder_button = Button(text="Select Folder")
        select_folder_button.bind(on_release=self.select_folder)
        layout.add_widget(select_folder_button)

        self.result_label = Label()
        layout.add_widget(self.result_label)

        return layout

    def load_labels(self, path):
        with open(path, "r") as file:
            return [line.strip() for line in file.readlines()]

    def update_folder_stats(self, folder_path):
        if os.path.exists(folder_path):
            self.total_count = 0
            self.correct_count = 0
            for root, _, files in os.walk(folder_path):
                for image_filename in files:
                    if image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(root, image_filename)
                        self.total_count += 1

                        input_details = self.model.get_input_details()
                        output_details = self.model.get_output_details()

                        image = self.load_and_preprocess_image(image_path, (320, 240))

                        self.model.set_tensor(input_details[0]["index"], image)
                        self.model.invoke()
                        predictions = self.model.get_tensor(output_details[0]["index"])

                        predicted_class = np.argmax(predictions)
                        predicted_label = self.labels[predicted_class]

                        true_label = os.path.basename(os.path.dirname(image_path))
                        if true_label == predicted_label:
                            self.correct_count += 1
        else:
            self.result_label.text = "Folder does not exist."

    def predict_images_in_folder(self):
        folder_path = self.folder_input.text
        self.update_folder_stats(folder_path)

        folder_result_label = Label()
        folder_result_label.text = f"Folder: {folder_path}, Total Images: {self.total_count}, Correct: {self.correct_count}, Incorrect: {self.total_count - self.correct_count}"
        self.result_label.text = folder_result_label.text

    def load_and_preprocess_image(self, image_path, target_size):
        image = PILImage.open(image_path).resize(target_size)
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        image = np.expand_dims(image, axis=0)
        return image

    def select_folder(self, instance):
        if platform == "android":
            filechooser.open_directory(on_selection=self.handle_folder_selection)
        else:
            self.result_label.text = "Folder selection is only available on Android."

    def handle_folder_selection(self, selection):
        if selection:
            folder_path = selection[0]
            self.folder_input.text = folder_path
            self.predict_images_in_folder()

if __name__ == "__main__":
    ImageClassifierApp().run()
