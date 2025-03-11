# Invoice OCR System

A robust solution for extracting structured data from invoice PDFs and images using AI vision models (both Google's Gemini and Anthropic's Claude).

## Project Overview

This system automates invoice data extraction by processing both PDF and image invoices, extracting key fields (dates, products, quantities, prices), applying markup calculations, and generating Excel outputs for analysis. The repository contains two implementations:

1. **Gemini Implementation** - Using Google's Gemini 2.0 Flash model
2. **Claude Implementation** - Using Anthropic's Claude 3.7 Sonnet model

Both implementations achieve similar results with different technical approaches and model-specific optimizations.

## Features

- Process invoices in both PDF and image formats (JPG, PNG, WEBP, HEIC, GIF)
- Extract structured data including:
  - Invoice dates
  - Product names and descriptions
  - Quantities
  - Unit prices
  - Total prices
- Apply a 25% markup on unit prices
- Generate individual Excel files per invoice
- Create a combined Excel file with all extracted data
- Robust error handling and retry logic
- Smart data cleaning to remove invalid/empty rows

## Development History

### Initial Approach with Gemini

Our development began with Google's Gemini API:
- Started with a basic implementation using the `gemini-pro-vision` model (later deprecated)
- Encountered and solved challenges with PDF processing by implementing direct PDF handling instead of conversion to images
- Added array normalization to handle mismatched data lengths in extracted JSON
- Enhanced data cleaning to improve output quality
- Fixed markup calculation to apply to unit price instead of total price

### Migration to Claude 3.7 Sonnet

To improve extraction quality, we implemented a version using Anthropic's Claude:
- Added support for both Anthropic's direct API and AWS Bedrock
- Implemented proper content type handling for different file types ("document" for PDFs, "image" for images)
- Enhanced media type detection for various formats
- Improved error handling for Claude's specific response structure
- Used system prompts to enhance extraction consistency

### Prompt Engineering Evolution

Throughout development, prompt engineering proved critical:
- Started with basic extraction requests
- Evolved to detailed, structured JSON extraction templates
- Added specific guidelines for consistent extraction between runs
- Refined to handle edge cases like preserving prefixes in product names

## Installation

### Requirements
- Python 3.8+
- Required packages depend on which implementation you use:

```bash
# For Gemini implementation
pip install google-generativeai pandas Pillow PyPDF2 openpyxl

# For Claude implementation
pip install anthropic pandas Pillow PyPDF2 openpyxl

# For Claude with AWS Bedrock
pip install boto3 pandas Pillow PyPDF2 openpyxl
```

## Usage

### Gemini Implementation

```python
# Configure the API key
import google.generativeai as genai
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Process invoices
from invoice_ocr_gemini import process_folder
process_folder(input_folder="/path/to/invoices", output_folder="output")
```

### Claude Implementation

```python
# For direct Anthropic API
import anthropic
anthropic_client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_API_KEY")

# Set configuration in the code:
# USE_ANTHROPIC_DIRECT = True
# USE_AWS_BEDROCK = False

# Process invoices
from invoice_ocr_claude import process_folder
process_folder(input_folder="/path/to/invoices", output_folder="output")
```

## Configuration Options

### Gemini Implementation
- Modify the prompt in `get_extraction_prompt()` for different extraction needs
- Adjust markup percentage by changing `df['Marked_Up_Price'] = df['U.Price'] * 1.25`
- Configure retry parameters in `call_gemini_with_retries()`

### Claude Implementation
- Choose between Anthropic API and AWS Bedrock by setting `USE_ANTHROPIC_DIRECT` or `USE_AWS_BEDROCK`
- Configure API credentials appropriate to your chosen method
- Adjust system prompt and extraction guidelines in `get_extraction_prompt()`
- Modify retry logic and exponential backoff in `call_claude_with_retries()`

## Technical Implementation Details

### File Processing
- PDFs are validated before processing using PyPDF2
- Images are processed with Pillow and converted to appropriate formats if needed
- Base64 encoding is used for sending files to AI models

### AI Model Integration
- Gemini: Uses `GenerativeModel` class with specific prompts
- Claude: Handles "document" and "image" content types differently
- Both implementations use retry logic with exponential backoff

### Data Extraction
- JSON parsing from model responses
- Array normalization to handle mismatched lengths
- Data type conversion for numeric fields
- Cleaning to remove invalid/empty rows

### Output Management
- Individual Excel files per invoice
- Combined Excel file with all data
- Sorting by date and product name

## Key Technical Insights

1. **Model-Specific Handling**
   - Gemini: Handles both PDFs and images similarly with minimal format differences
   - Claude: Requires different content types for PDFs ("document") and images ("image")

2. **Error Handling Best Practices**
   - Validation before processing prevents downstream errors
   - Retry logic with exponential backoff overcomes transient API issues
   - Response format validation ensures data integrity

3. **Prompt Engineering Impact**
   - Clear, specific guidelines substantially improve extraction consistency
   - Examples of edge cases help models handle difficult inputs
   - Format instructions ensure parseable outputs

4. **Performance Considerations**
   - Claude excels with complex layouts and has native PDF understanding
   - Gemini offers good integration with Google ecosystem
   - Both models handle most invoice formats effectively with proper prompting

## Limitations and Considerations

- Claude 3.7 Sonnet has a 100-page limit for PDFs
- Very large files may require timeout adjustments
- Some complex table structures might need custom extraction logic
- Model outputs can vary slightly between runs, even with identical inputs
- API costs vary based on model choice and usage volume

## Future Improvements

- Implement a hybrid approach using both models for verification
- Add extraction confidence scores
- Implement invoice template recognition for enhanced accuracy
- Add a web interface for easier use

---

This project demonstrates the power of modern AI vision models for document processing automation, with proper engineering making the difference between experimental prototypes and production-ready systems.

---
