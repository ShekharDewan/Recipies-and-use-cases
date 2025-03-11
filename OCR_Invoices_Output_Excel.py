import os
import base64
import json
import re
import time
import datetime
from collections import Counter
import pandas as pd
from PIL import Image
from pathlib import Path
from PyPDF2 import PdfReader
from io import BytesIO

# Choose ONE of these import blocks based on your preferred access method:

# Option 1: Using Anthropic's direct API
import anthropic

# Option 2: Using AWS Bedrock
import boto3
from botocore.config import Config

# Configure which client to use (set only one to True)
USE_ANTHROPIC_DIRECT = True
USE_AWS_BEDROCK = False

# Configure clients based on choice
if USE_ANTHROPIC_DIRECT:
    # Initialize Anthropic client
    anthropic_client = anthropic.Anthropic(api_key="YOUR-API-KEY-HERE")
    MODEL_ID = "claude-3-7-sonnet-20250219"
elif USE_AWS_BEDROCK:
    # Initialize AWS Bedrock client
    bedrock_config = Config(
        region_name="us-east-1",  # Change to your preferred region
        signature_version="v4",
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    bedrock_runtime = boto3.client('bedrock-runtime', config=bedrock_config)
    MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"
else:
    raise ValueError("You must enable either Anthropic API or AWS Bedrock")

def validate_pdf(pdf_path):
    """Validate PDF using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        print(f"PDF {pdf_path} has {page_count} pages")
        # Claude 3.7 Sonnet has a limit of 100 pages per PDF
        if page_count > 100:
            print(f"Warning: PDF has {page_count} pages, exceeding Claude's 100-page limit")
        return True
    except Exception as e:
        print(f"Invalid PDF {pdf_path}: {e}")
        return False

def extract_json_from_response(response_text):
    """Extract valid JSON from the response text."""
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None
    
    try:
        json_data = json.loads(json_match.group())
        # Verify we have at least product data
        if 'Product' not in json_data or not json_data['Product']:
            return None
        return json_data
    except json.JSONDecodeError:
        return None

def ensure_equal_length_arrays(json_data):
    """Ensure all arrays in the JSON have equal length, with smarter padding."""
    array_keys = [key for key, value in json_data.items() if isinstance(value, list)]
    if not array_keys:
        return json_data
    
    # Find non-empty arrays and their lengths
    non_empty_lengths = [len(json_data[key]) for key in array_keys if len(json_data[key]) > 0]
    if not non_empty_lengths:
        return json_data
    
    # Use the most common non-zero length
    length_counts = Counter(non_empty_lengths)
    target_length = length_counts.most_common(1)[0][0]
    
    # Adjust arrays to the target length
    for key in array_keys:
        current_length = len(json_data[key])
        if current_length == 0:
            # For empty arrays, create appropriate placeholders
            filler = 0 if key in ['Qty', 'U.Price', 'Total'] else ''
            json_data[key] = [filler] * target_length
        elif current_length < target_length:
            # For shorter arrays, add padding
            filler = 0 if key in ['Qty', 'U.Price', 'Total'] else ''
            json_data[key].extend([filler] * (target_length - current_length))
        elif current_length > target_length:
            # For longer arrays, truncate to avoid excessive padding
            json_data[key] = json_data[key][:target_length]
    
    return json_data

def clean_dataframe(df):
    """Remove empty or mostly zero rows from the extracted data."""
    # Create a mask for rows to keep
    # Keep rows where Product has a value or Total is greater than 0
    mask = (df['Product'].str.strip() != '') & (df['Product'].notna())
    mask = mask | ((df['Total'] > 0) & df['Total'].notna())
    
    # Apply the mask to keep only valid rows
    cleaned_df = df[mask].copy()
    
    # Reset the index
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def call_claude_with_retries(prompt_text, content, is_pdf=False, retries=3, delay=5):
    """Call Claude 3.7 Sonnet API with retry logic."""
    for attempt in range(retries):
        try:
            if USE_ANTHROPIC_DIRECT:
                # Using Anthropic API directly
                message = anthropic_client.messages.create(
                    model=MODEL_ID,
                    max_tokens=8192,
                    # Use system prompt to disable thinking block for now
                    # since we only need the final result
                    system="Please provide only the final extraction result without thinking blocks.",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                content
                            ]
                        }
                    ]
                )
                
                # For regular responses without thinking blocks
                if hasattr(message.content[0], 'text'):
                    return message.content[0].text
                else:
                    # Try to find any text content
                    for block in message.content:
                        if hasattr(block, 'text'):
                            return block.text
                    
                    # If no text content is found
                    raise ValueError("No text content found in Claude's response")
            
            elif USE_AWS_BEDROCK:
                # Using AWS Bedrock
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 8192,
                    "system": "Please provide only the final extraction result without thinking blocks.",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                content
                            ]
                        }
                    ]
                }
                
                response = bedrock_runtime.invoke_model(
                    modelId=MODEL_ID,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read().decode('utf-8'))
                
                # Extract text content from the response
                if 'content' in response_body and len(response_body['content']) > 0:
                    for content_block in response_body['content']:
                        if 'type' in content_block and content_block['type'] == 'text':
                            return content_block['text']
                
                raise ValueError("No text content found in Claude's response")
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:  # Don't sleep after the last attempt
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    raise Exception("All retries failed")

def get_extraction_prompt():
    """Return the extraction prompt for invoice data."""
    return """
    You are a specialized invoice data extraction system. Extract the following information from this invoice document:

    EXTRACTION GUIDELINES:

    1. INVOICE DATE
       - Format: YYYY-MM-DD 
       - Location: Usually at the top of the invoice
       - Extract only the actual invoice date, not order or shipping dates
       - If multiple dates appear, prioritize the one labeled "Invoice Date" or "Date"

    2. PRODUCT INFORMATION
       - Extract each product as a separate line item
       - Capture the COMPLETE product name exactly as written, including:
         * ALL prefixes (like "BD2", "BIA", etc.)
         * ALL details (weight, size, origin country, etc.)
         * Maintain original capitalization and punctuation
       - Examples: "BD2 Coriander Eng" (not just "Coriander Eng"), "Kenya Long Ravaiya -- 4kg"

    3. QUANTITY
       - Extract the exact quantity for each product
       - Format as a number without units
       - If quantity is not explicitly stated, leave it blank/null (do NOT default to 1)

    4. UNIT PRICE
       - Extract the price per individual unit
       - Format as a decimal number without currency symbols
       - Example: 10.50 (not £10.50 or $10.50)

    5. TOTAL PRICE
       - Extract the total price for each line item
       - Format as a decimal number without currency symbols

    IMPORTANT NOTES:
    - Extract ONLY product line items (ignore subtotals, tax lines, shipping fees, etc.)
    - Preserve exact text as it appears (don't "fix" typos or standardize names)
    - Be especially careful with product prefixes like "BD2", "BIA" - include them every time
    - When the same product appears in different invoices, extract it consistently

    Your output MUST be formatted as a JSON object with the following structure:
    {
      "Date": ["2023-06-24", "2023-06-24", ...],  
      "Product": ["BD2 Coriander Eng", "BIA MINT Eng", ...],
      "Qty": [50, 10, ...],
      "U.Price": [7.5, 7.0, ...],
      "Total": [375, 70, ...]
    }
    """

def process_pdf(pdf_path):
    """Process PDF file with Claude 3.7 Sonnet."""
    try:
        if not validate_pdf(pdf_path):
            return pd.DataFrame()
            
        # For Claude 3.7 Sonnet, PDFs must be sent as "document" type, not "image" type
        if USE_ANTHROPIC_DIRECT:
            with open(pdf_path, 'rb') as f:
                pdf_content = {
                    "type": "document",  # IMPORTANT: Use "document" not "image" for PDFs
                    "source": {
                        "type": "base64", 
                        "media_type": "application/pdf",
                        "data": base64.b64encode(f.read()).decode('utf-8')
                    }
                }
        elif USE_AWS_BEDROCK:
            # AWS Bedrock also requires "document" type for PDFs
            with open(pdf_path, 'rb') as f:
                pdf_content = {
                    "type": "document",  # IMPORTANT: Use "document" not "image" for PDFs
                    "source": {
                        "type": "base64", 
                        "media_type": "application/pdf",
                        "data": base64.b64encode(f.read()).decode('utf-8')
                    }
                }
        
        # Call Claude with the PDF
        prompt = get_extraction_prompt()
        response = call_claude_with_retries(prompt, pdf_content, is_pdf=True)
        
        # Extract and process JSON
        json_data = extract_json_from_response(response)
        if not json_data:
            print(f"No valid JSON found in response for {pdf_path}")
            return pd.DataFrame()
        
        # Normalize and create DataFrame
        json_data = ensure_equal_length_arrays(json_data)
        df = pd.DataFrame(json_data)
        
        # Convert numeric columns
        for col in ['Qty', 'U.Price', 'Total']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate markup on unit price (not total)
        if 'U.Price' in df.columns:
            df['Marked_Up_Price'] = df['U.Price'] * 1.25
        
        # Add source information and clean the data
        df['Source'] = os.path.basename(pdf_path)
        df = clean_dataframe(df)
        
        return df
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return pd.DataFrame()

def process_image(image_path):
    """Process image file with Claude 3.7 Sonnet."""
    try:
        # Open the image and convert to base64
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format if img.format else 'JPEG')
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Determine media type based on image format
            if img.format and img.format.lower() in ['jpeg', 'jpg']:
                media_type = 'image/jpeg'
            elif img.format and img.format.lower() == 'png':
                media_type = 'image/png'
            elif img.format and img.format.lower() == 'webp':
                media_type = 'image/webp'
            elif img.format and img.format.lower() == 'gif':
                media_type = 'image/gif'
            else:
                # Convert to PNG for other formats
                buffered = BytesIO()
                img.save(buffered, format='PNG')
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                media_type = 'image/png'
        
        # Prepare image content for Claude - use "image" type for images
        if USE_ANTHROPIC_DIRECT:
            image_content = {
                "type": "image",  # IMPORTANT: Use "image" for images
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_str
                }
            }
        elif USE_AWS_BEDROCK:
            image_content = {
                "type": "image",  # IMPORTANT: Use "image" for images
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_str
                }
            }
        
        # Call Claude with the image
        prompt = get_extraction_prompt()
        response = call_claude_with_retries(prompt, image_content, is_pdf=False)
        
        # Extract and process JSON
        json_data = extract_json_from_response(response)
        if not json_data:
            print(f"No valid JSON found in response for {image_path}")
            return pd.DataFrame()
        
        # Normalize and create DataFrame
        json_data = ensure_equal_length_arrays(json_data)
        df = pd.DataFrame(json_data)
        
        # Convert numeric columns
        for col in ['Qty', 'U.Price', 'Total']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate markup on unit price (not total)
        if 'U.Price' in df.columns:
            df['Marked_Up_Price'] = df['U.Price'] * 1.25
        
        # Add source information and clean the data
        df['Source'] = os.path.basename(image_path)
        df = clean_dataframe(df)
        
        return df
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return pd.DataFrame()

def process_folder(input_folder="/content/invoices", output_folder="output"):
    """Process all invoices in folder and save results."""
    os.makedirs(output_folder, exist_ok=True)
    all_data = []
    
    # Get list of files to process
    files = os.listdir(input_folder)
    total_files = len(files)
    print(f"Found {total_files} files in {input_folder}")
    
    # Process files
    processed_count = 0
    for file in files:
        file_path = os.path.join(input_folder, file)
        if os.path.isdir(file_path):
            continue
            
        processed_count += 1
        print(f"Processing file {processed_count}/{total_files}: {file}")
            
        if file.lower().endswith(('.pdf', '.PDF')):
            df = process_pdf(file_path)
        elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.heic', '.gif')):
            df = process_image(file_path)
        else:
            print(f"Skipping unsupported file: {file}")
            continue
            
        if not df.empty:
            # Save individual file
            output_path = os.path.join(output_folder, f"{Path(file).stem}.xlsx")
            df.to_excel(output_path, index=False)
            print(f"Saved data to {output_path}")
            all_data.append(df)
        else:
            print(f"No data extracted from {file}")
    
    # Save combined results
    if all_data:
        # Combine all individual DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Apply final cleaning to remove any remaining empty/zero rows
        combined_df = clean_dataframe(combined_df)
        
        # Sort by Date and then by Product for better organization
        if 'Date' in combined_df.columns:
            combined_df.sort_values(['Date', 'Product'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        
        # Generate timestamp and save
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = os.path.join(output_folder, f"combined_invoices_{timestamp}.xlsx")
        combined_df.to_excel(combined_path, index=False)
        print(f"\nProcessed {len(all_data)} files successfully. Combined file saved to {combined_path}")
        print(f"Combined file contains {len(combined_df)} rows of data after cleaning.")
    else:
        print("No data extracted from any files. Combined file not created.")

if __name__ == "__main__":
    process_folder()
