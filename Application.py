# Importing Libraries

import numpy as np
import cv2
import os, sys
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime

# from hunspell import Hunspell
# import enchant

from tensorflow.keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self):

        # self.hs = Hunspell('en_US')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # File upload variables
        self.video_file = None
        self.video_cap = None
        self.is_processing_video = False
        self.video_queue = queue.Queue()
        
        # Animation variables
        self.animation_counter = 0
        self.pulse_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.current_color_index = 0
        
        self.json_file = open("Models/model_new_fixed_v2.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Models/model_new_fixed_v2.weights.h5")

        self.json_file_dru = open("Models/model-bw_dru_fixed.json" , "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()

        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("Models/model-bw_dru_fixed.weights.h5")
        self.json_file_tkdi = open("Models/model-bw_tkdi_fixed.json" , "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()

        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("Models/model-bw_tkdi_fixed.weights.h5")
        self.json_file_smn = open("Models/model-bw_smn_fixed.json" , "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()

        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("Models/model-bw_smn_fixed.weights.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("🤟 Sign Language To Text Conversion 🤟")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        # Enhanced GUI with file upload and animations
        self.root.geometry("1200x1000")
        self.root.configure(bg='#2C3E50')

        self.panel = tk.Label(self.root)
        self.panel.place(x = 120, y = 60, width = 640, height = 480)
        
        self.panel2 = tk.Label(self.root) # Processed image panel
        self.panel2.place(x = 780, y = 60, width = 200, height = 200)

        # Title with animation
        self.T = tk.Label(self.root)
        self.T.place(x = 300, y = 10)
        self.T.config(text = "🤟 Sign Language To Text Conversion 🤟", 
                    font = ("Arial", 28, "bold"), fg="white", bg='#2C3E50')
        
        # File upload section
        self.upload_frame = tk.Frame(self.root, bg='#34495E', relief=tk.RAISED, bd=2)
        self.upload_frame.place(x = 50, y = 570, width = 1100, height = 80)
        
        self.upload_btn = tk.Button(self.upload_frame, text="📁 Upload Video File", 
                                  command=self.upload_video, 
                                  font=("Arial", 14, "bold"),
                                  bg='#3498DB', fg='white',
                                  padx=20, pady=10)
        self.upload_btn.place(x = 20, y = 20)
        
        self.process_btn = tk.Button(self.upload_frame, text="▶️ Process Video", 
                                   command=self.process_video_file,
                                   font=("Arial", 14, "bold"),
                                   bg='#27AE60', fg='white',
                                   padx=20, pady=10,
                                   state=tk.DISABLED)
        self.process_btn.place(x = 200, y = 20)
        
        self.file_label = tk.Label(self.upload_frame, text="No file selected", 
                                 font=("Arial", 12), bg='#34495E', fg='white')
        self.file_label.place(x = 380, y = 25)
        
        self.progress_label = tk.Label(self.upload_frame, text="", 
                                     font=("Arial", 12), bg='#34495E', fg='#E74C3C')
        self.progress_label.place(x = 800, y = 25)

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 540)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350, y = 645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 645)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250, y = 690)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 26, y = 745)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 325, y = 745)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 625, y = 745)


        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        # Start animation
        self.animate_ui()
        self.video_loop()


    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            # Dynamic ROI based on frame center
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 3
            x1 = w // 2 - roi_size // 2
            y1 = h // 2 - roi_size // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 2)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)

            # Extract ROI with error checking
            if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                roi = cv2image[y1:y2, x1:x2]
                
                # Enhanced preprocessing pipeline
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Noise reduction
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                
                # Adaptive threshold with better parameters
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
                
                # Morphological operations for cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Edge enhancement
                edges = cv2.Canny(binary, 50, 150)
                binary = cv2.addWeighted(binary, 0.7, edges, 0.3, 0)
                
                self.predict(binary)

                self.current_image2 = Image.fromarray(binary)
                imgtk2 = ImageTk.PhotoImage(image = self.current_image2)
                self.panel2.imgtk = imgtk2
                self.panel2.config(image = imgtk2)

            self.panel3.config(text = self.current_symbol, font = ("Courier", 30))

            self.panel4.config(text = self.word, font = ("Courier", 30))

            self.panel5.config(text = self.str,font = ("Courier", 30))

            # predicts = self.hs.suggest(self.word)
            predicts = []  # Disabled hunspell
            
            if(len(predicts) > 1):

                self.bt1.config(text = predicts[0], font = ("Courier", 20))

            else:

                self.bt1.config(text = "")

            if(len(predicts) > 2):

                self.bt2.config(text = predicts[1], font = ("Courier", 20))

            else:

                self.bt2.config(text = "")

            if(len(predicts) > 3):

                self.bt3.config(text = predicts[2], font = ("Courier", 20))

            else:

                self.bt3.config(text = "")


        self.root.after(10, self.video_loop)  # Slightly slower for stability

    def predict(self, test_image):
        
        try:
            # Enhanced preprocessing pipeline
            if test_image.size == 0:
                return
                
            test_image = cv2.resize(test_image, (128, 128))
            
            # Multi-stage enhancement
            # 1. Contrast enhancement
            test_image = cv2.equalizeHist(test_image)
            
            # 2. Normalization
            test_image = test_image.astype('float32') / 255.0
            
            # 3. Add slight noise for robustness
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.02, test_image.shape)
                test_image = np.clip(test_image + noise, 0, 1)
            
            # 4. Prepare for model
            test_image = np.expand_dims(test_image, axis=-1)
            
            # Get prediction with error handling
            result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
            
            # Build comprehensive prediction dictionary
            prediction = {}
            prediction['blank'] = float(result[0][0])
            
            # Map all letters A-Z
            for inde, letter in enumerate(ascii_uppercase):
                prediction[letter] = float(result[0][inde + 1])
            
            # Apply ensemble-like prediction smoothing
            # Temperature scaling for better distribution
            temperature = 1.2
            for key in prediction:
                prediction[key] = np.exp(np.log(max(prediction[key], 1e-8)) / temperature)
            
            # Normalize
            total = sum(prediction.values())
            if total > 0:
                for key in prediction:
                    prediction[key] /= total
            
            # Sort and select prediction
            prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
            self.current_symbol = prediction[0][0]
            
            # Enhanced debug output
            top_predictions = prediction[:5]
            print(f"Top 5: {[(char, f'{prob:.3f}') for char, prob in top_predictions]}")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.current_symbol = 'blank'
            return

        #LAYER 2 - Disabled for now (specialized models have random weights)
        # The original multi-layer classification is disabled for this demo
        
        if(self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if(self.ct[self.current_symbol] > 60):

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':

                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word

                    self.word = ""

        
        # Enhanced debug output
        top_predictions = prediction[:5]
        print(f"Top 5: {[(char, f'{prob:.3f}') for char, prob in top_predictions]}")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        self.current_symbol = 'blank'
        return

    #LAYER 2 - Disabled for now (specialized models have random weights)
    # The original multi-layer classification is disabled for this demo
    
    if(self.current_symbol == 'blank'):

        for i in ascii_uppercase:
            self.ct[i] = 0

    self.ct[self.current_symbol] += 1

    if(self.ct[self.current_symbol] > 60):

        for i in ascii_uppercase:
            if i == self.current_symbol:
                continue

            tmp = self.ct[self.current_symbol] - self.ct[i]

            if tmp < 0:
                tmp *= -1

            if tmp <= 20:
                self.ct['blank'] = 0

                for i in ascii_uppercase:
                    self.ct[i] = 0
                return

        self.ct['blank'] = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        if self.current_symbol == 'blank':

            if self.blank_flag == 0:
                self.blank_flag = 1

                if len(self.str) > 0:
                    self.str += " "

                self.str += self.word

                self.word = ""

        else:

            if(len(self.str) > 16):
                self.str = ""

            self.blank_flag = 0

            self.word += self.current_symbol

def animate_ui(self):
    """Animate UI elements for better user experience"""
    self.animation_counter += 1
    
    # Pulse animation for title
    if self.animation_counter % 20 == 0:
        self.current_color_index = (self.current_color_index + 1) % len(self.pulse_colors)
        color = self.pulse_colors[self.current_color_index]
        self.T.config(fg=color)
    
    # Update displays with current data
    if hasattr(self, 'char_display'):
        self.char_display.config(text=self.current_symbol if self.current_symbol != 'Empty' else '-')
        self.word_display.config(text=self.word)
        self.sentence_display.config(text=self.str)
    
    # Continue animation
    self.root.after(100, self.animate_ui)

def action3(self):
    pass

def destructor(self):

    print("Closing Application...")

    self.root.destroy()
    self.vs.release()
    cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()