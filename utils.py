import re
import html


def convert_to_html_body(text):
    # Escape HTML entities to prevent potential security issues
    escaped_text = html.escape(text)

    # Split the text into paragraphs based on line breaks
    paragraphs = escaped_text.split('\n\n')

    # Wrap each paragraph in <p> tags and join them together
    html_paragraphs = '\n\n'.join(f'<p>{paragraph}</p>' for paragraph in paragraphs)

    # Replace single line breaks with <br> tags within each paragraph
    html_body = html_paragraphs.replace('\n\n', '<p></p>')

    return html_body

def obfuscate_data(text):
    obfuscated_text = text
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',    # <- Email
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'                              # <- Credit card
    ]

    for pattern in patterns:
        obfuscated_text = re.sub(pattern, '< REDACTED_PERSONAL_INFORMATION >', text)

    return obfuscated_text

# Read a txt file
def read_file(file):
    with open(file, 'r') as f:
        return f.read().split('\n\n')



# x = read_file('./data/faqs_life.txt')
# print(x)