import logging

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class PDFDocumentConverter:
    """A class to convert PDF documents to markdown with formula enrichment."""

    def __init__(self, enable_formula_enrichment=True, log_level=logging.INFO):
        """
        Initialize the PDF converter with pipeline options.

        Args:
            enable_formula_enrichment (bool): Enable formula enrichment in PDFs
            log_level: Logging level (default: logging.INFO)
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Initializing PDFDocumentConverter")

        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_formula_enrichment = enable_formula_enrichment
        self.logger.debug(f"Formula enrichment: {enable_formula_enrichment}")

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        self.logger.info("DocumentConverter initialized successfully")

    def convert_to_markdown(self, file_path):
        """
        Convert a PDF file to markdown format.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            str: Markdown representation of the document
        """
        try:
            self.logger.info(f"Starting conversion of: {file_path}")
            result = self.converter.convert(source=file_path)
            markdown = result.document.export_to_markdown()
            self.logger.info(f"Successfully converted {file_path} to markdown")
            self.logger.debug(f"Markdown length: {len(markdown)} characters")
            return markdown
        except Exception as e:
            self.logger.error(f"Error converting {file_path}: {str(e)}")
            raise

    def get_metadata(self, file_path):
        """
        Extract metadata from a PDF file.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            dict: PDF metadata
        """
        try:
            self.logger.info(f"Extracting metadata from: {file_path}")
            doc = pymupdf.open(file_path)
            metadata = doc.metadata
            self.logger.info(f"Successfully extracted metadata from {file_path}")
            self.logger.debug(f"Metadata keys: {list(metadata.keys())}")
            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            raise

    def process_document(self, file_path):
        """
        Process a PDF document: extract metadata and convert to markdown.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            tuple: (metadata, markdown_content)
        """
        metadata = self.get_metadata(file_path)
        markdown = self.convert_to_markdown(file_path)
        return metadata, markdown

    def print_document_info(self, file_path):
        """
        Print metadata and markdown content of a PDF document.

        Args:
            file_path (str): Path to the PDF file
        """
        metadata, markdown = self.process_document(file_path)
        print(metadata)
        print(markdown)


# Usage example
if __name__ == "__main__":
    converter = PDFDocumentConverter()
    file_path = input("Enter a file path: ")
    converter.print_document_info(file_path)
