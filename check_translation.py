import PyPDF2

def extract_text_from_pdf(pdf_path, max_pages=5):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(f"Total pages in PDF: {num_pages}")
        for page_num in range(min(max_pages, num_pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            print(f"\n--- Page {page_num + 1} ---\n{text}")

if __name__ == "__main__":
    translated_pdf = "Guidebook_용접 공정최적화 AI 데이터셋_RU.pdf"
    extract_text_from_pdf(translated_pdf) 