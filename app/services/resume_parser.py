import pdfplumber
import docx
import io
import re

class ResumeParser:
    @staticmethod
    def extract_text(file_content: bytes, filename: str) -> str:
        """
        Extracts text from PDF or DOCX file content.
        """
        text = ""
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = ResumeParser._extract_from_pdf(file_content)
        elif lower.endswith(".docx"):
            text = ResumeParser._extract_from_docx(file_content)
        elif lower.endswith(".txt"):
            text = file_content.decode("utf-8", errors="replace")
        else:
            raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")
        
        return ResumeParser._clean_text(text)

    @staticmethod
    def _extract_from_pdf(file_content: bytes) -> str:
        text = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    @staticmethod
    def _extract_from_docx(file_content: bytes) -> str:
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    @staticmethod
    def _clean_text(text: str) -> str:
        # Remove extra whitespace and non-printable characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Basic non-ascii removal, valid for English resumes
        return text.strip()
