"""
Tests for extractor_wrapper.
Input: List of dictionaries with the following keys: url, domain, raw_html.
Output: List of dictionaries with the following keys: company_name, SNI, text, phone_number, org_number, address, municipality, postal_code, url.
It uses the pytest library to run the tests. To run simply execute the following command in the terminal:
> pytest tests/test_extract_wrapper.py
"""
import requests
from pathlib import Path
from scripts.extract_wrapper import extract_wrapper
import pytest
import tldextract
import json

DATASET = [
            {
                "company_name": "SSAB",
                "SNI": "123456789",
                "text": "",
                "phone_number": "123456789",
                "org_number": "123456789",
                "address": "123456789",
                "municipality": "Stockgolm",
                "postal_code": "123456789",
                "url": "https://ssab.se/"
            },
            {
                "company_name": "LKAB",
                "SNI": "123456789",
                "text": "",
                "phone_number": "123456789",
                "org_number": "123456789",
                "address": "123456789",
                "municipality": "Stockgolm",
                "postal_code": "123456789",
                "url": "https://lkab.com/"
            }
        ]

@pytest.fixture
def urls():
    """
    Return a list of website urls to test against.
    """
    return ['https://ssab.se/','https://www.ssab.com/sv-se/kontakt', 'https://lkab.com/']

@pytest.fixture
def temp_scraped_data(tmp_path, urls):
    """
    Create a temporary input file containing a list of dictionaries with website information.
    The file has the same structure as the output from the scraper.
    """
    lst = []
    for url in urls:
        lst.append({'domain': tldextract.extract(url).domain, 'url': url, 'raw_html': requests.get(url).text})
    
    input_file = tmp_path / "input_scraped_data.json"
    with open(Path(input_file), 'w', encoding='utf-8') as f:
        json.dump(lst, f, indent=4, ensure_ascii=False)
        return input_file
    
@pytest.fixture
def temp_dataset(tmp_path):
    """
    Create a temporary input file containing a list of dictionaries with website information.
    The file has the same structure as the dataset.
    """
    input_file = tmp_path / "input_data.json"
    with open(Path(input_file), 'w', encoding='utf-8') as f:
        json.dump(DATASET, f, indent=4, ensure_ascii=False)
        return input_file
    

def test_structure(tmp_path, temp_scraped_data, temp_dataset):
    """
    Test the structure of the output file generated by the extractor_wrapper function.
    """
    temp_output_file = tmp_path / "output.json"
    extract_wrapper(temp_dataset, temp_scraped_data, temp_output_file, extract_meta=True, extract_body=False, p_only=False)
    
    with open(temp_output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert isinstance(item, dict)
        assert 'company_name' in item and isinstance(item['company_name'], str)
        assert 'SNI' in item and isinstance(item['SNI'], str)
        assert 'text' in item and isinstance(item['text'], str)
        assert 'phone_number' in item and isinstance(item['phone_number'], str)
        assert 'org_number' in item and isinstance(item['org_number'], str)
        assert 'address' in item and isinstance(item['address'], str)
        assert 'municipality' in item and isinstance(item['municipality'], str)
        assert 'postal_code' in item and isinstance(item['postal_code'], str)
        assert 'url' in item and isinstance(item['url'], str)
    
    
    
    temp_scraped_data.unlink() # delete the temp file
    temp_dataset.unlink() # delete the temp file
    temp_output_file.unlink() # delete the temp file
    