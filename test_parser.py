import os
from resume_parser import ResumeParser

def test_resume_parser():
    # Initialize the parser
    parser = ResumeParser()
    
    # Check if API key is set
    if not parser.api_key:
        print("Error: LLAMAPARSE_API_KEY environment variable is not set")
        return
    
    # Test file path - you'll need to provide a sample resume PDF
    test_file = "sample_resume.pdf"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found")
        return
    
    # Try parsing the resume
    try:
        result = parser.parse_resume(test_file)
        if result:
            print("Successfully parsed resume!")
            print("\nParsed Content:")
            print("-" * 50)
            print(f"Raw Content Length: {len(result['raw_content'])} characters")
            print("\nMetadata:")
            for key, value in result['metadata'].items():
                print(f"{key}: {value}")
            print("\nSections:")
            for section, content in result['sections'].items():
                print(f"\n{section.upper()}:")
                print(content)
        else:
            print("Failed to parse resume")
    except Exception as e:
        print(f"Error during parsing: {str(e)}")

if __name__ == "__main__":
    test_resume_parser() 