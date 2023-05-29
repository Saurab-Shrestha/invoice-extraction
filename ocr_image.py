import numpy as np
import pandas as pd
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path


def invoice_to_text(file_path, page_numbers=None):
    file_extension = file_path.split(".")[-1]
    if file_extension.lower() == 'pdf':
        # Convert PDF to images
        images = convert_from_path(file_path)
        pages_text = []
        
        if page_numbers is None:
            page_numbers = range(1, len(images) + 1)

        for i, image in enumerate(images, start=1):
            if i in page_numbers:
                # Convert image to grayscale
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

                # Perform text extraction on the image
                text = extract_text(gray)
                pages_text.append(text)

        return '\n\n'.join(pages_text)
    
    else:
        # Read the image file
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform text extraction on the image
        text = extract_text(gray)
        return text
    
    
def extract_text(image):
    # Apply preprocessing to enhance the text extraction
    enhanced_image = preprocess_image(image)

    # Perform OCR using Tesseract
    custom_config = r'-l eng --oem 1 --psm 6'
    d = pytesseract.image_to_data(enhanced_image, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)

    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()

    text = ""
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]

        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0

        for ix, ln in curr.iterrows():
            # Add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # Number of spaces that should be added
            if ln['left'] / char_w > prev_left + 1:
                added = int(ln['left'] / char_w) - prev_left
                text += ' ' * added

            text += ln['text'] + '_'
            prev_left += len(ln['text']) + added + 1

    return text


def preprocess_image(image):
    # Apply histogram equalization for enhancing the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    return enhanced_image


def image_or_pdf_to_excel(file_path, page_numbers=None):
    text = invoice_to_text(file_path, page_numbers)
    print(text)

    lines = text.strip().split('\n')
    data = [line.split() for line in lines]
    # Combine split values in the same row
    # combined_data = []
    # for row in data:
    #     combined_row = []
    #     for i in range(len(row)):
    #         if i > 0 and row[i-1].endswith(','):  # Check if the previous value ended with a comma
    #             combined_row[-1] += ' ' + row[i]  # Combine the current value with the previous one
    #         else:
    #             combined_row.append(row[i])
    #     combined_data.append(combined_row)

    # df = pd.DataFrame(combined_data)
    df = pd.DataFrame(data)
    df = df.replace("_"," ", regex=True)
    # df.to_excel("data_invoice_1.xlsx", header=False, index=False) # uncomment if you want the file in excel
    return df



if __name__ == "__main__":
    file_path = "../"   # give image or pdf file path.
    page_numbers = []   # give page number for pdf file, Default = None
    df = image_or_pdf_to_excel(file_path, page_numbers) # the returns Dataframe

