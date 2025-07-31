import cv2
import pytesseract
import numpy as np
import os
import csv

# Configuration
INPUT_IMAGE = "screenshots/test.jpg"
OUTPUT_CSV = "ptcgp_collection.csv"
TESSERACT_PATH = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def detect_cards(image):
    """
    Find all card regions in screenshot.
    Assumes cards are of relatively uniform size and aspect ratio,
    and includes visual debugging for thresholded image and detected contours.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle varying lighting conditions across the screenshot.
    # blockSize: Should be an odd number (e.g., 11, 15, 21). Larger values for smoother results,
    #             smaller for finer details.
    # C: Constant subtracted from the mean. Adjust to make text clearer (positive for darker, negative for lighter).
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 2) 
    
    # --- DEBUGGING STEP 1: Visualize the thresholded image ---
    cv2.imshow('1. Thresholded Image for Contour Detection (Press any key to continue)', thresh)
    cv2.waitKey(0) 

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cards = []
    print("\n--- Analyzing Found Contours ---")
    
    # Create a copy of the original image to draw on for debugging detected cards
    debug_image = image.copy() 

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        # --- DEBUGGING: Print dimensions of every contour found ---
        print(f"  Contour {i+1}: x={x}, y={y}, W={w}, H={h}, Aspect Ratio={aspect_ratio:.2f}")
        
        # --- CRITICAL TUNING PARAMETERS ---
        # Adjust these values based on your screenshot's card pixel dimensions and aspect ratio.
        # Use the '2. All Contours' window to refine these.
        
        # Minimum width and height (pixels) for a detected card.
        # If actual cards are smaller, reduce these. If non-card objects are detected, increase.
        MIN_CARD_WIDTH = 150 
        MIN_CARD_HEIGHT = 200 
        
        # Acceptable aspect ratio range for a standard PTCG card (typically around 0.7 for portrait)
        # Widen this range if cards are slightly angled or have minor perspective distortion.
        MIN_ASPECT_RATIO = 0.65 
        MAX_ASPECT_RATIO = 0.75 
        
        if (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and 
            w > MIN_CARD_WIDTH and h > MIN_CARD_HEIGHT):
            
            cards.append((x, y, w, h))
            # Draw a green rectangle around the detected card on the debug image
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"  --> Identified as a Card (Passes criteria)!")
        else:
            # Draw a red rectangle for contours that were ignored
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            print(f"  --> Ignored (Did not pass criteria)")

    # --- DEBUGGING STEP 2: Visualize all contours found, highlighting detected cards ---
    cv2.imshow('2. All Contours (Green=Detected, Red=Ignored) (Press any key to continue)', debug_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    return cards

def extract_card_details(card_img):
    """
    Optimized OCR for PTCGP card names and attempts to get set/number.
    Assumes consistent placement of name and set details within each card crop.
    Includes visual debugging for the ROI images fed into Tesseract.
    """
    h, w, _ = card_img.shape

    # --- CRITICAL TUNING PARAMETERS: REGIONS OF INTEREST (ROIs) ---
    # These coordinates are percentages of the detected card's height (h) and width (w).
    # You MUST adjust these by looking at your actual 'Card X Crop' and the subsequent
    # 'Name ROI' and 'Set/Number ROI' debug windows to precisely frame the text.
    
    # Card Name Region: Usually at the top, centered horizontally.
    # [Y_start:Y_end, X_start:X_end]
    NAME_ROI = card_img[int(h*0.05):int(h*0.35), int(w*0.05):int(w*0.95)] 
    
    # Set Symbol / Card Number Region: Typically at the bottom-right.
    SET_NUMBER_ROI = card_img[int(h*0.80):int(h*0.95), int(w*0.70):int(w*0.95)] 

    # --- OCR for Card Name ---
    gray_name = cv2.cvtColor(NAME_ROI, cv2.COLOR_BGR2GRAY)
    thresh_name = cv2.threshold(gray_name, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # --- DEBUGGING STEP 4: Display the thresholded Name ROI for OCR ---
    cv2.imshow('4. Name ROI for OCR (Press any key to continue)', thresh_name)
    cv2.waitKey(0) 

    # Tesseract configuration for name (PSM 7: Assume a single text line)
    # Whitelist commonly found characters in card names, including spaces, hyphens, slashes, periods, parentheses, ampersands, exclamation marks.
    config_name = r'--psm 7 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/.()&! " '
    card_name = pytesseract.image_to_string(thresh_name, config=config_name).strip()
    
    # --- OCR for Set and Number ---
    gray_set_num = cv2.cvtColor(SET_NUMBER_ROI, cv2.COLOR_BGR2GRAY)
    thresh_set_num = cv2.threshold(gray_set_num, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # --- DEBUGGING STEP 5: Display the thresholded Set/Number ROI for OCR ---
    cv2.imshow('5. Set/Number ROI for OCR (Press any key to continue)', thresh_set_num)
    cv2.waitKey(0)

    # Tesseract config for set/number (PSM 7, whitelist for numbers, slash, common set code characters)
    config_set_num = r'--psm 7 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"'
    set_number_raw = pytesseract.image_to_string(thresh_set_num, config=config_set_num).strip()

    set_info = "UNKNOWN_SET"
    card_number = "UNKNOWN_NUMBER"

    # Basic parsing for set_number_raw (e.g., "123/150", "SWSH")
    if '/' in set_number_raw:
        parts = set_number_raw.split('/')
        if len(parts) >= 1: 
            card_number = parts[0].strip()
            # If the raw string contains a slash, we'll use the full string for set_info for now,
            # as it might include the set number and total count, or set code.
            set_info = set_number_raw 
    elif set_number_raw: 
        # If no slash, maybe it's just the set code (e.g., "SWSH", "FST")
        set_info = set_number_raw

    cv2.destroyAllWindows() 

    return card_name, set_info, card_number

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}. Ensure it exists and is a valid image file.")
        return []
    
    print(f"--- Processing image: {image_path} ---")
    # Pass a copy of the image to detect_cards so it can draw on it without affecting subsequent operations
    cards = detect_cards(img.copy()) 
    print(f"\nFound {len(cards)} potential card regions.")
    
    collection = []
    if not cards:
        print("No card regions detected based on current parameters.")
        return []

    print("\n--- Extracting details from detected cards ---")
    for i, (x, y, w, h) in enumerate(cards):
        card_img = img[y:y+h, x:x+w]
        
        # --- DEBUGGING STEP 3: Display the individual cropped card image ---
        cv2.imshow(f'3. Card {i+1} Crop (Press any key to continue)', card_img)
        cv2.waitKey(0) 

        name, set_info, card_number = extract_card_details(card_img)
        
        if name or set_info: 
            collection.append([name, set_info, card_number, 1])  # [Card Name, Set Info, Card Number, Quantity]
            print(f"  Processed Card {i+1}: Name='{name}', Set='{set_info}', Number='{card_number}'")
        else:
            print(f"  Processed Card {i+1}: No identifiable text detected within defined ROIs.")
            
    return collection

if __name__ == "__main__":
    os.makedirs("screenshots", exist_ok=True)
    
    if not os.path.exists(INPUT_IMAGE):
        print(f"Input image not found: {INPUT_IMAGE}")
        print(f"Please ensure '{INPUT_IMAGE}' exists in the 'screenshots' folder and contains clear images of PTCGP cards.")
        print(f"You can capture one using: adb exec-out screencap -p > {INPUT_IMAGE}")
    else:
        cards_collected = process_image(INPUT_IMAGE)
        
        if cards_collected:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Card Name", "Set Info", "Card Number", "Quantity"])
                writer.writerows(cards_collected)
            print(f"\n--- Successfully saved {len(cards_collected)} cards to {OUTPUT_CSV} ---")
        else:
            print("\n--- Finished processing: No cards were added to the collection. ---")
            print("To improve detection and extraction, please:")
            print("  1. Carefully review the debug windows (1-5) at each step.")
            print("  2. Adjust the `MIN_CARD_WIDTH`, `MIN_CARD_HEIGHT`, `MIN_ASPECT_RATIO`, `MAX_ASPECT_RATIO` in `detect_cards`.")
            print("  3. Adjust the percentage-based ROIs (`NAME_ROI`, `SET_NUMBER_ROI`) in `extract_card_details`.")
            print("  4. Ensure your screenshot is well-lit and clear.")
