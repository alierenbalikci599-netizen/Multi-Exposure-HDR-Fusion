import cv2
import numpy as np

def run_hdr_pipeline():
    # Load images
    file_paths = ['short.jpg', 'normal.jpg', 'long.jpg']
    exposure_times = np.array([1/60.0, 1/15.0, 1/4.0], dtype=np.float32)
    
    img_list = [cv2.imread(f) for f in file_paths]
    if None in img_list:
        print("Error: Images not found.")
        return

    # Align
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(img_list, img_list)

    # Mertens + CLAHE (Our Solution)
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')

    lab = cv2.cvtColor(res_mertens_8bit, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    res_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # Save Output
    cv2.imwrite("final_output.jpg", res_clahe)
    print("Success: final_output.jpg has been generated.")

if __name__ == "__main__":
    run_hdr_pipeline()
