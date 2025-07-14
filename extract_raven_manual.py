import PyPDF2

with open('RavenManual_v3.8.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    all_text = ''
    for page in reader.pages:
        all_text += page.extract_text() + '\n'

with open('RavenManual_v3.8.txt', 'w', encoding='utf-8') as out:
    out.write(all_text)

print('Extracted text to RavenManual_v3.8.txt') 