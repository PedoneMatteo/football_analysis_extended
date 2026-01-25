# Bilanciamento Luci
# Maschera Colore 
# Rilevamento Bordi 
# Regressione Matematica.
import numpy as np 
import cv2
# -------------------------------------------------------------------------------------------------------------------------------
 
            ####################################
            ##       BILANCIAMENTO LUCI       ##
            ####################################

# Per "pulire" l'immagine e rendere le linee d'erba visibili ovunque, 
# la tecnica migliore è l'uso del CLAHE (Contrast Limited Adaptive Histogram Equalization).

# Come funziona il Bilanciamento Adattivo (CLAHE)
# A differenza di un bilanciamento normale (che schiarisce tutto il frame allo stesso modo), 
# il CLAHE divide l'immagine in una griglia (chiamata tileGridSize) e ottimizza il contrasto cella per cella.
# 
# Distribuzione locale: 
# Se una cella è molto scura (ombra a destra), il CLAHE ne espande i valori per renderla visibile. 
# Se è molto chiara (sole a sinistra), cerca di recuperare i dettagli.
# 
# Limite del rumore: 
# Il "Contrast Limited" impedisce all'algoritmo di esagerare, 
# evitando che il rumore digitale diventi troppo evidente nelle zone scure.

def preprocess_image(image):
    # 1. Convertiamo da BGR (colori standard) a LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Dividiamo i canali: L (Luce), A (Verde-Rosso), B (Blu-Giallo)
    l, a, b = cv2.split(lab)
    
    # 3. Creiamo l'oggetto CLAHE
    # clipLimit: quanto deve essere "aggressivo" il contrasto (2.0 - 5.0 è l'ideale)
    # tileGridSize: la dimensione della griglia di analisi
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    
    # 4. Applichiamo solo al canale della luce (L)
    l_balanced = clahe.apply(l)
    
    # 5. Riuniamo i canali e torniamo in BGR per le fasi successive
    lab_final = cv2.merge((l_balanced, a, b))
    final_image = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    return final_image

# -------------------------------------------------------------------------------------------------------------------------------

            ###################################
            ##        MASCHERA COLORE        ##
            ###################################

# Ora che abbiamo un'immagine con una luminosità bilanciata grazie al CLAHE, possiamo passare alla segmentazione del colore.
# 
# L'obiettivo è isolare i due toni di verde (erba chiara e erba scura) 
# per individuare il punto esatto in cui si toccano: quel "confine" è la nostra linea del campo.
# 
# 1. Perché usare lo spazio colore HSV?
# Invece del classico RGB (Rosso, Verde, Blu), per la maschera colore usiamo HSV (Hue, Saturation, Value). 
# È molto più efficace perché separa l'informazione del colore da quella della luce:
# - Hue (Tonalità): Il tipo di colore (es. "Verde"). Resta simile sia al sole che all'ombra.
# - Saturation (Saturazione): Quanto è intenso il colore.
# - Value (Valore): La luminosità.
# 
# In RGB, se un'ombra cade sul prato, cambiano tutti e tre i valori (R, G, B). 
# In HSV, cambierà principalmente il Value, rendendo molto più semplice dire al computer: 
# "Prendi tutto ciò che è verde, a prescindere da quanto sia illuminato".

def get_grass_masks(balanced_image):
    hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
    
    # --- MASCHERA CHIARA (Light) ---
    # Alziamo la saturazione minima (da 40 a 60) per ignorare il verde sbiadito
    # Alziamo il valore minimo (da 100 a 150) per prendere solo il "brillante"
    lower_light = np.array([35, 60, 140]) 
    upper_light = np.array([55, 255, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    # --- MASCHERA SCURA (Dark) ---
    # Restringiamo il valore massimo (da 100 a 140) per non sovrapporci troppo al chiaro
    lower_dark = np.array([35, 30, 20])
    upper_dark = np.array([60, 255, 100])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
  # --- PULIZIA MORFOLOGICA POTENZIATA ---

    # 1. Kernel verticale molto più alto per "cucire" i buchi distanti
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 40))

    # 2. Applica una DILATAZIONE prima del closing (opzionale ma efficace)
    # Questo espande il bianco per "mangiare" le macchie nere interne
    mask_dark = cv2.dilate(mask_dark, np.ones((3,3), np.uint8), iterations=1)

    # Closing: riempie i buchi (giocatori, texture erba)
    mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel_v)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel_v)
    
    # Opening: elimina piccoli puntini isolati (rumore negli spalti)
    kernel_small = np.ones((4,4), np.uint8)
    mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_OPEN, kernel_small)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel_small)

    return mask_light, mask_dark
# -------------------------------------------------------------------------------------------------------------------------------

            ###################################
            ##    DISEGNO LINEE SUL PRATO    ##
            ###################################

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

# -------------------------------------------------------------------------------------------------------------------------------
            
            ###################################
            ##             MAIN              ##
            ###################################

if __name__ == "__main__":
    # Carica un'immagine di esempio
    image = cv2.imread('./input_videos/match.png')

    # 1) Bilanciamento luci
    preprocess_imaged = preprocess_image(image)

    # 2) Maschera colore
    mask_light, mask_dark = get_grass_masks(preprocess_imaged)
    # Uniamo le maschere per vedere la copertura totale
    combined_grass = cv2.bitwise_or(mask_light, mask_dark)

    # Estraiamo i bordi specifici della maschera chiara (le tue strisce)
    edges_light = cv2.Canny(mask_light, 100, 200)

    # Visualizza questo per capire se le linee sono dritte
    cv2.imwrite('output_videos/edges_test.png', edges_light)

    cv2.imwrite('output_videos/combined_mask.png', combined_grass)
    cv2.imwrite('output_videos/mask_light.png', mask_light)
    cv2.imwrite('output_videos/mask_dark.png', mask_dark)
    #output_image = draw_grass_lines(image, grass_lines)
    
    # Mostra l'immagine risultante
    cv2.imwrite('output_videos/grass_lines_detected.png', preprocess_imaged)