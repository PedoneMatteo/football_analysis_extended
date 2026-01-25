import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Raffinamento della Maschera HSV
# Invece di una maschera "passa-tutto" per il verde, dobbiamo creare due maschere specifiche 
# per i due toni di erba  e poi trovare il confine dove si toccano.
def get_grass_transition_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Verde Chiaro
    lower_light = np.array([35, 40, 100])
    upper_light = np.array([50, 255, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    # Verde Scuro
    lower_dark = np.array([35, 40, 30])
    upper_dark = np.array([55, 255, 100])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Uniamo le maschere per isolare solo il prato (escludendo spalti e cartelloni)
    grass_mask = cv2.bitwise_or(mask_light, mask_dark)
    
    # Applichiamo un Blur per pulire il rumore dei fili d'erba
    grass_mask = cv2.GaussianBlur(grass_mask, (7, 7), 0)
    
    return grass_mask, mask_light, mask_dark

# bordo tra le zone chiare e scure.
def find_grass_cuts(mask_light):
    # Usiamo un kernel verticale perché le linee che cerchiamo sono verticali in prospettiva
    kernel = np.ones((1, 5), np.uint8)
    
    # Trova i bordi della zona "chiara"
    edge = cv2.morphologyEx(mask_light, cv2.MORPH_GRADIENT, kernel)
    
    # Pulizia finale
    _, edge = cv2.threshold(edge, 10, 255, cv2.THRESH_BINARY)
    return edge

def detect_refined_grass_lines(frame):
    # 1. Isola il prato
    grass_mask, mask_l, mask_d = get_grass_transition_mask(frame)
    
    # 2. Trova i bordi tra i due verdi
    edges = find_grass_cuts(mask_l)
    
    # 3. HoughLinesP con parametri specifici per linee lunghe
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,      # Abbassato per catturare linee meno marcate
        minLineLength=150, # Solo linee lunghe (le strisce)
        maxLineGap=30      # Unisce segmenti interrotti dai giocatori
    )
    return lines

def extrapolate_line_point(x1, y1, x2, y2, target_y):
    """Calcola la X di una linea data una certa Y (anche fuori frame)"""

    if x2 - x1 == 0: return x1 # Linea verticale
     # Linea orizzontale (o quasi)
    if abs(y2 - y1) < 1:
        return None
    
    m = (y2 - y1) / (x2 - x1) # Pendenza
    # Formula: y - y1 = m(x - x1)  => x = (y - y1)/m + x1
    target_x = (target_y - y1) / m + x1
    return int(target_x)

ANGLE_EPS = 7.0  # gradi (3–7 funziona bene)

# Filtrare per Angolo e Posizione
def filter_grass_lines(lines, img_height):
    candidates = []
    if lines is None: return candidates

    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        
        # Calcoliamo l'angolo della linea
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        
        # Le linee dell'erba in prospettiva sono quasi verticali (es. tra 60 e 120 gradi)
        if 30 < angle < 120:
            # Filtriamo anche in base alla lunghezza per evitare piccoli segmenti rumorosi
            #length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            #if length > img_height / 3: 
            candidates.append((angle,line.flatten()))

    candidates.sort(key=lambda x: x[0]) 
    filtered = []
    last_angle = None

    for angle, line in candidates:
        if last_angle is None or abs(angle - last_angle) > ANGLE_EPS:
            print(f"Detected line with angle: {angle:.2f} degrees")
            filtered.append(line)
            last_angle = angle
        # else: linea scartata perché angolo troppo simile

    return filtered    

def draw_grass_lines(image, lines):
    if lines is None:
        return image

    height, width, _ = image.shape
    for line in lines:
        x1, y1, x2, y2 = line
        # Estendere la linea fino ai bordi superiore e inferiore dell'immagine
        top_x = extrapolate_line_point(x1, y1, x2, y2, 0)
        bottom_x = extrapolate_line_point(x1, y1, x2, y2, height - 1)
        if top_x is None or bottom_x is None:
            continue
        cv2.line(image, (top_x, 0), (bottom_x, height - 1), (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    # Carica un'immagine di esempio
    image = cv2.imread('./input_videos/match.png')
    
    # Rileva le linee delle strisce d'erba
    lines = detect_refined_grass_lines(image)
    grass_lines = filter_grass_lines(lines, image.shape[0])
    #extreme_lines = get_extreme_lines(grass_lines)
    # Disegna le linee rilevate sull'immagine
    output_image = draw_grass_lines(image, grass_lines)
    
    # Mostra l'immagine risultante
    cv2.imwrite('output_videos/grass_lines_detected.png', output_image)
