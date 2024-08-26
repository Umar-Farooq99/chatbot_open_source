import streamlit as st
from modules.pdf_extraction import extract_text_from_pdf, PDFExtractionError
from modules.document_search import create_document_search, DocumentSearchError
from modules.llm_response import get_response_from_llm, LLMResponseError

def main():
    # Set Streamlit page config
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    pdf = st.file_uploader("Upload your PDF here", type='pdf')
    if pdf is not None:
        try:
            text = extract_text_from_pdf(pdf)
            doc_search = create_document_search(text)
            user_question = st.text_input("Ask a question about your PDF")

            if user_question:
                response = get_response_from_llm(doc_search, user_question)
                st.write(response)
        except PDFExtractionError as e:
            st.error(str(e))
        except DocumentSearchError as e:
            st.error(str(e))
        except LLMResponseError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
