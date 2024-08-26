from PyPDF2 import PdfReader

class PDFExtractionError(Exception):
    pass

def extract_text_from_pdf(pdf):
    try:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise PDFExtractionError(f"Error extracting text from PDF: {str(e)}")