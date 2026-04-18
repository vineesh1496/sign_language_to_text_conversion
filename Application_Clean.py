# Importing Libraries

import numpy as np
import cv2
import os
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
import pickle

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

class Application:
    """Clean Sign Language Recognition Application with 65% Accuracy Models"""

    def __init__(self):
        # Camera setup
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # Load trained 65% accuracy models
        print("🚀 Loading Trained Models...")
        
        if os.path.exists('fast_90_percent_models.pkl'):
            with open('fast_90_percent_models.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.trained_ensemble = model_data['ensemble_model']
                self.trained_scaler = model_data['scaler']
                self.model_accuracy = model_data['accuracy']
                print(f"✅ Loaded {self.model_accuracy*100:.1f}% accuracy models!")
        else:
            print("❌ Trained models not found!")
            self.trained_ensemble = None
            self.trained_scaler = None
            self.model_accuracy = 0.0
        
        # Variables
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Animation variables
        self.animation_counter = 0
        self.pulse_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.current_color_index = 0

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title(f"🎯 {self.model_accuracy*100:.0f}% Accuracy Sign Language Recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1000x700")
        self.root.configure(bg='#2C3E50')

        # Create GUI elements
        self.create_gui()

        # Start video loop
        self.video_loop()

    def create_gui(self):
        """Create clean GUI layout"""
        # Main panels
        self.panel = tk.Label(self.root, bg='#34495E', relief=tk.SUNKEN, bd=3)
        self.panel.place(x=50, y=60, width=640, height=480)
        
        self.panel2 = tk.Label(self.root, bg='#34495E', relief=tk.SUNKEN, bd=3)
        self.panel2.place(x=720, y=60, width=200, height=200)

        # Title with animation
        self.T = tk.Label(self.root)
        self.T.place(x=200, y=10)
        self.T.config(text=f"🎯 {self.model_accuracy*100:.0f}% Accuracy Sign Language Recognition", 
                    font=("Arial", 24, "bold"), fg="white", bg='#2C3E50')
        
        # Labels
        self.video_label = tk.Label(self.root, text="📹 Live Camera Feed", 
                                font=("Arial", 14, "bold"), fg="white", bg='#2C3E50')
        self.video_label.place(x=250, y=540)

        self.symbol_label = tk.Label(self.root, text="🤟 Detected Symbol:", 
                                font=("Arial", 14, "bold"), fg="white", bg='#2C3E50')
        self.symbol_label.place(x=720, y=280)

        # Current symbol display
        self.current_symbol = "Empty"
        self.symbol = tk.Label(self.root, textvariable=self.current_symbol, 
                             font=("Arial", 36, "bold"), fg="#FF6B6B", bg='#34495E')
        self.symbol.place(x=720, y=310)

        # Word display
        self.word_label = tk.Label(self.root, text="📝 Word:", 
                                font=("Arial", 14, "bold"), fg="white", bg='#2C3E50')
        self.word_label.place(x=720, y=400)

        self.word = tk.Label(self.root, text="", font=("Arial", 20, "bold"), 
                           fg="#4ECDC4", bg='#34495E')
        self.word.place(x=720, y=430)

        # Sentence display
        self.sentence_label = tk.Label(self.root, text="📄 Sentence:", 
                                   font=("Arial", 14, "bold"), fg="white", bg='#2C3E50')
        self.sentence_label.place(x=720, y=490)

        self.sentence = tk.Label(self.root, text="", font=("Arial", 16, "bold"), 
                              fg="#96CEB4", bg='#34495E', wraplength=180)
        self.sentence.place(x=720, y=520)

        # Control buttons
        self.clear_btn = tk.Button(self.root, text="🗑️ Clear", 
                              command=self.clear_all,
                              font=("Arial", 12, "bold"),
                              bg='#E74C3C', fg='white',
                              relief=tk.RAISED, bd=2)
        self.clear_btn.place(x=50, y=620, width=100, height=40)

        self.save_btn = tk.Button(self.root, text="💾 Save", 
                             command=self.save_results,
                             font=("Arial", 12, "bold"),
                             bg='#27AE60', fg='white',
                             relief=tk.RAISED, bd=2)
        self.save_btn.place(x=170, y=620, width=100, height=40)

        # Initialize text variables
        self.str = ""
        self.word_text = ""
        self.current_symbol = "Empty"

    def video_loop(self):
        """Main video processing loop"""
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)

            # Define ROI
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[0])

            cv2.rectangle(cv2image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            # Preprocess
            binary = cv2.adaptiveThreshold(cv2image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            binary = cv2.bilateralFilter(binary, 9, 75, 75)
            
            roi = binary[y1:y2, x1:x2]
            self.current_image = cv2image
            self.current_image2 = binary

            # Predict
            self.predict(roi)

            # Update GUI
            self.update_gui()

        self.root.after(1, self.video_loop)

    def predict(self, test_image):
        """Predict using trained models"""
        try:
            if test_image.size == 0 or not self.trained_ensemble:
                return
                
            # Preprocess
            test_image = cv2.resize(test_image, (128, 128))
            test_image = cv2.equalizeHist(test_image)
            test_image = test_image.astype('float32') / 255.0

            # Extract features
            features = self.extract_features(test_image)
            feature_vector = self.features_to_vector(features)
            
            # Scale features
            if self.trained_scaler:
                feature_vector = self.trained_scaler.transform([feature_vector])
            else:
                feature_vector = feature_vector.reshape(1, -1)

            # Predict
            prediction = self.trained_ensemble.predict(feature_vector)[0]
            confidence = max(self.trained_ensemble.predict_proba(feature_vector)[0])
            
            self.current_symbol = prediction
            
            # Debug output
            print(f"🎯 Prediction: {prediction} (Conf: {confidence:.3f})")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.current_symbol = 'blank'

    def extract_features(self, image):
        """Extract features for trained model"""
        # Convert to uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(image)
        features['std'] = np.std(image)
        features['var'] = np.var(image)
        features['min'] = np.min(image)
        features['max'] = np.max(image)
        features['range'] = features['max'] - features['min']
        
        # Pixel distribution
        white_pixels = np.sum(image > 127)
        total_pixels = image.size
        features['white_ratio'] = white_pixels / total_pixels
        features['black_ratio'] = 1 - features['white_ratio']
        features['edge_ratio'] = np.sum(cv2.Canny(image, 50, 150) > 0) / total_pixels
        features['gray_ratio'] = np.sum((image > 50) & (image < 200)) / total_pixels
        
        # Spatial distribution
        h, w = image.shape
        h_half, w_half = h // 2, w // 2
        
        quadrants = [
            image[:h_half, :w_half],  # TL
            image[:h_half, w_half:],  # TR
            image[h_half:, :w_half],  # BL
            image[h_half:, w_half:]   # BR
        ]
        
        for i, quad in enumerate(quadrants):
            quad_white = np.sum(quad > 127)
            features[f'quad_{i}_ratio'] = quad_white / quad.size
        
        # Contour features
        try:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                features['contour_area'] = cv2.contourArea(largest) / total_pixels
                features['contour_perimeter'] = cv2.arcLength(largest, True) / max(h, w)
                
                if features['contour_perimeter'] > 0:
                    features['compactness'] = 4 * np.pi * features['contour_area'] / (features['contour_perimeter'] ** 2)
                else:
                    features['compactness'] = 0
                
                # Bounding box
                x, y, w_box, h_box = cv2.boundingRect(largest)
                features['bbox_aspect'] = w_box / max(h_box, 1)
                features['bbox_fill'] = cv2.contourArea(largest) / (w_box * h_box)
            else:
                features.update({
                    'contour_area': 0, 'contour_perimeter': 0, 'compactness': 0,
                    'bbox_aspect': 1, 'bbox_fill': 0
                })
        except:
            features.update({
                'contour_area': 0.1, 'contour_perimeter': 0.1, 'compactness': 0.1,
                'bbox_aspect': 1, 'bbox_fill': 0.1
            })
        
        # Center of mass
        white_coords = np.where(image > 127)
        if len(white_coords[0]) > 0:
            features['center_y'] = np.mean(white_coords[0]) / h
            features['center_x'] = np.mean(white_coords[1]) / w
        else:
            features['center_y'] = 0.5
            features['center_x'] = 0.5
        
        return features

    def features_to_vector(self, features):
        """Convert features to vector"""
        feature_order = [
            'mean', 'std', 'var', 'min', 'max', 'range',
            'white_ratio', 'black_ratio', 'edge_ratio',
            'quad_0_ratio', 'quad_1_ratio', 'quad_2_ratio', 'quad_3_ratio',
            'contour_area', 'contour_perimeter', 'compactness',
            'bbox_aspect', 'bbox_fill',
            'center_y', 'center_x',
            'gray_ratio'  # Add missing feature to make 21 total
        ]
        
        return np.array([features.get(f, 0) for f in feature_order])

    def update_gui(self):
        """Update GUI with animations"""
        try:
            # Update camera feed
            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)
            
            # Update processed image
            img2 = cv2.cvtColor(self.current_image2, cv2.COLOR_GRAY2RGB)
            img2 = Image.fromarray(img2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            self.panel2.imgtk2 = imgtk2
            self.panel2.configure(image=imgtk2)
            
            # Animate title
            self.animation_counter += 1
            if self.animation_counter % 10 == 0:
                self.current_color_index = (self.current_color_index + 1) % len(self.pulse_colors)
                color = self.pulse_colors[self.current_color_index]
                self.T.config(fg=color)
            
            # Update text displays
            self.update_text_displays()
            
        except Exception as e:
            print(f"GUI update error: {e}")

    def update_text_displays(self):
        """Update text displays"""
        if self.current_symbol != "blank":
            if self.blank_flag == 0:
                if self.current_symbol.lower() >= 'a' and self.current_symbol.lower() <= 'z':
                    self.str += self.current_symbol.lower()
                    self.word_text += self.current_symbol.lower()
                    
                    if len(self.word_text) >= 5:
                        self.word_text = self.word_text + " "
                        self.sentence.config(text=self.sentence.cget("text") + " " + self.word_text)
                        self.word_text = ""
        else:
            self.blank_flag = 0

        self.word.config(text=self.word_text)

    def clear_all(self):
        """Clear all results"""
        self.str = ""
        self.word_text = ""
        self.current_symbol = "Empty"
        self.sentence.config(text="")

    def save_results(self):
        """Save results to file"""
        content = f"Sign Language Recognition Results\n"
        content += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"Accuracy: {self.model_accuracy*100:.1f}%\n"
        content += f"Current sentence: {self.sentence.cget('text')}\n"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(content)
            messagebox.showinfo("Saved", f"Results saved to:\n{file_path}")

    def destructor(self):
        """Clean up"""
        print("Closing Application...")
        if self.vs.isOpened():
            self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    print("🚀 Starting Clean 65% Accuracy Sign Language Application...")
    app = Application()
    app.root.mainloop()
