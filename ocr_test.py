import cv2
import pytesseract
import numpy as np
import os
import csv
from datetime import datetime

# Configuration
INPUT_IMAGE = "screenshots/test.jpg"
OUTPUT_CSV = "ptcgp_collection.csv"
DEBUG_DIR = "debug_steps"
TESSERACT_PATH = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def debug_show(img, name="Debug", scale=0.5, save=True):
    """Display and optionally save debug images"""
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    
    if save:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S")
        cv2.imwrite(f"{DEBUG_DIR}/{timestamp}_{name}.png", img)
    
    cv2.imshow(name, resized)
    cv2.waitKey(100)  # Brief pause to see images
    return resized

def extract_card_details(card_img, card_num):
    """Enhanced extraction with visual debugging"""
    h, w = card_img.shape[:2]
    debug_show(card_img, f"card_{card_num}_original", save=True)
    
    # 1. Preprocess entire card
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    debug_show(gray, f"card_{card_num}_gray", save=True)
    
    # 2. Extract NAME (top section)
    name_region = card_img[int(h*0.01):int(h*0.15), int(w*0.1):int(w*0.9)]
    debug_show(name_region, f"card_{card_num}_name_region", save=True)
    
    name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    debug_show(name_thresh, f"card_{card_num}_name_thresh", save=True)
    
    name_text = pytesseract.image_to_string(
        name_thresh,
        config='--psm 7 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ "'
    ).strip()
    
    # 3. Extract DETAILS (bottom section)
    details_region = card_img[int(h*0.7):int(h*0.9), int(w*0.1):int(w*0.9)]
    debug_show(details_region, f"card_{card_num}_details_region", save=True)
    
    details_gray = cv2.cvtColor(details_region, cv2.COLOR_BGR2GRAY)
    _, details_thresh = cv2.threshold(details_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    debug_show(details_thresh, f"card_{card_num}_details_thresh", save=True)
    
    details_text = pytesseract.image_to_string(
        details_thresh,
        config='--psm 6 --oem 3'
    ).strip()
    
    return name_text, details_text

def process_image(image_path):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return []
    
    debug_show(img, "original_image", save=True)
    
    # 1. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    debug_show(gray, "gray_image", save=True)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    debug_show(blur, "blurred_image", save=True)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    debug_show(thresh, "threshold_image", save=True)
    
    # 2. Find cards
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} potential card regions")
    
    # Visualize all contours
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
    debug_show(contour_img, "all_contours", save=True)
    
    collection = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Card size filters (adjust these!)
        if 100 < w < 400 and 150 < h < 600:
            card_img = img[y:y+h, x:x+w]
            
            # Draw bounding box
            box_img = img.copy()
            cv2.rectangle(box_img, (x,y), (x+w,y+h), (0,0,255), 2)
            debug_show(box_img, f"card_{i}_box", save=True)
            
            name, details = extract_card_details(card_img, i)
            
            if name:
                print(f"Card {i}: {name}")
                collection.append([name, details])
    
    return collection

if __name__ == "__main__":
    if not os.path.exists(INPUT_IMAGE):
        print(f"First capture a screenshot:\nadb exec-out screencap -p > {INPUT_IMAGE}")
    else:
        cards = process_image(INPUT_IMAGE)
        
        if cards:
            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Card Name", "Details"])
                writer.writerows(cards)
            print(f"\nSaved {len(cards)} cards to {OUTPUT_CSV}")
        else:
            print("\nNo cards detected. Please check:")
            print(f"1. Debug images in {DEBUG_DIR}/")
            print("2. Adjust these parameters in process_image():")
            print("   - Card size filters (100 < w < 400 and 150 < h < 600)")
            print("3. Try different threshold values in extract_card_details()")
    
    cv2.destroyAllWindows()
