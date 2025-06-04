import PyPDF2
from deep_translator import GoogleTranslator
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def translate_pdf(input_pdf, output_pdf):
    try:
        # Initialize translator
        translator = GoogleTranslator(source='ko', target='ru')
        logger.info("Translator initialized")
        
        # Read the PDF file
        logger.info(f"Reading PDF file: {input_pdf}")
        with open(input_pdf, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            # Create a new PDF with translated text
            packet = BytesIO()
            c = canvas.Canvas(packet, pagesize=letter)
            
            # Use local DejaVuSans.ttf font for Russian support
            font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
                c.setFont("DejaVuSans", 12)
                logger.info("DejaVuSans font loaded from local file.")
            else:
                logger.warning("DejaVuSans.ttf not found in project directory, falling back to Helvetica")
                c.setFont("Helvetica", 12)
            
            # Process each page
            for page_num in range(num_pages):
                logger.info(f"Processing page {page_num + 1}/{num_pages}")
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if not text.strip():
                    logger.warning(f"No text found on page {page_num + 1}")
                    continue
                
                # Clean and translate the text
                try:
                    cleaned_text = clean_text(text)
                    translated = translator.translate(cleaned_text)
                    logger.info(f"Translated page {page_num + 1}")
                except Exception as e:
                    logger.error(f"Translation error on page {page_num + 1}: {str(e)}")
                    continue
                
                # Add translated text to the new PDF
                y = 750  # Starting y position
                for line in translated.split('\n'):
                    if line.strip():
                        # Split long lines to fit page width
                        words = line.split()
                        current_line = []
                        for word in words:
                            if len(' '.join(current_line + [word])) < 80:  # Adjust line length as needed
                                current_line.append(word)
                            else:
                                c.drawString(50, y, ' '.join(current_line))
                                y -= 15
                                current_line = [word]
                        if current_line:
                            c.drawString(50, y, ' '.join(current_line))
                            y -= 15
                
                c.showPage()
            
            c.save()
            logger.info("Canvas saved")
            
            # Move to the beginning of the StringIO buffer
            packet.seek(0)
            
            # Create a new PDF with the translated content
            new_pdf = PyPDF2.PdfWriter()
            new_pdf.add_page(PyPDF2.PdfReader(packet).pages[0])
            
            # Save the new PDF
            with open(output_pdf, 'wb') as output_file:
                new_pdf.write(output_file)
            logger.info(f"Translated PDF saved to: {output_pdf}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = "Guidebook_용접 공정최적화 AI 데이터셋.pdf"
    output_file = "Guidebook_용접 공정최적화 AI 데이터셋_RU.pdf"
    translate_pdf(input_file, output_file) 