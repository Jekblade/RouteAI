import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR values of the clicked pixel
        b, g, r = params['image'][y, x]
        
        # Print clicked pixel's colors for debugging
        print(f"Clicked pixel (B, G, R): ({b}, {g}, {r})")
        
        # Calculate the correction factor for each channel
        # Avoid division by zero
        correction_factor = [255.0 / channel if channel != 0 else 1 for channel in (b, g, r)]
        
        # Apply the correction factor to each channel
        corrected_image = np.zeros_like(params['image'], dtype=np.float32)
        for i in range(3):  # B, G, R
            corrected_image[:, :, i] = params['image'][:, :, i] * correction_factor[i]
        
        # Clip values to [0, 255] and convert back to uint8
        corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
        
        # Display the corrected image
        cv2.imshow('Corrected Image', corrected_image)
        
        # Optionally, save the corrected image
        cv2.imwrite(params['output_path'], corrected_image)

def apply_white_balance_with_manual_selection(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    
    # Show the original image and wait for a click
    cv2.imshow('Original Image', image)
    params = {'image': image, 'output_path': output_image_path}
    cv2.setMouseCallback('Original Image', click_event, params)
    
    # Wait until any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = 'path/to/your/input/image.png'
output_image_path = 'path/to/your/output/image_corrected.png'
apply_white_balance_with_manual_selection(input_image_path, output_image_path)
