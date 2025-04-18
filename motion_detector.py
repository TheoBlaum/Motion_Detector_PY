import cv2
import numpy as np
import time
import threading
from queue import Queue
from datetime import datetime  # Ajout de l'import pour l'horodatage

class MotionFaceDetector:
    def __init__(self):
        # Configuration
        self.motion_threshold = 1000  # Seuil de détection de mouvement
        self.face_scale_factor = 1.1  # Facteur d'échelle pour la détection de visages
        self.face_min_neighbors = 5   # Nombre minimum de voisins pour la détection de visages
        self.face_min_size = (30, 30) # Taille minimale des visages
        self.blur_kernel = (21, 21)   # Taille du noyau de flou
        
        # Initialisation
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.prev_blurred = None  # Initialisation de prev_blurred
        
    def start_camera(self):
        # Essayer différentes sources de caméra
        for camera_index in [0, 1, -1]:
            print(f"Essai de la caméra {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print(f"Impossible d'ouvrir la caméra {camera_index}")
                self.cap.release()
                continue
                
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"Impossible de lire depuis la caméra {camera_index}")
                self.cap.release()
                continue
                
            print(f"Caméra {camera_index} connectée avec succès!")
            # Initialiser prev_blurred avec la première frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            return True
            
        return False
    
    def capture_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
    
    def process_frame(self, frame):
        # Inverser l'image horizontalement
        frame = cv2.flip(frame, 1)
        
        # Convertir en niveaux de gris une seule fois
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Détection de visages
        faces = self.face_cascade.detectMultiScale(
            gray,  # Utiliser l'image en niveaux de gris déjà calculée
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=self.face_min_size
        )
        
        # Dessiner les rectangles des visages
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Obtenir la date et l'heure actuelles
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Afficher les informations
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, current_time, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame, blurred
    
    def run(self):
        if not self.start_camera():
            print("Aucune caméra n'a pu être ouverte")
            return
        
        self.running = True
        capture_thread = threading.Thread(target=self.capture_thread)
        capture_thread.start()
        
        while self.running:
            # Calculer les FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
            
            # Récupérer une nouvelle frame
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Traiter la frame
                processed_frame, curr_blurred = self.process_frame(frame)
                
                # Détection de mouvement
                if self.prev_blurred is not None:  # Vérifier que prev_blurred existe
                    diff = cv2.absdiff(self.prev_blurred, curr_blurred)
                    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < self.motion_threshold:
                            continue
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Afficher la frame
                cv2.imshow('Motion and Face Detection', processed_frame)
                
                # Mettre à jour l'image précédente
                self.prev_blurred = curr_blurred
            
            # Vérifier la touche 'q' pour quitter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Nettoyage
        self.running = False
        capture_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MotionFaceDetector()
    detector.run() 