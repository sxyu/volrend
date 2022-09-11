# Combine all files into index.html for simple distribution
import sys
from os import path
import base64

OUTPUT_HTML = sys.argv[1]
MAIN_HTML = sys.argv[2]
MAIN_JS = sys.argv[3]
MAIN_WASM = sys.argv[4]

# Inline the WASM into the js
with open(MAIN_WASM, 'rb') as f:
    wasm = f.read()
    wasm = base64.b64encode(wasm).decode('utf-8')

with open(MAIN_JS, 'r') as f:
    main_js = f.read()
    main_js = main_js.replace("{{inline_wasm_b64}}", wasm)

# Inline the JS into the html, to ultimately produce only 1 file
with open(MAIN_HTML, 'r') as f:
    html = f.read()
    html = html.replace('{{main_js}}', main_js)

with open(OUTPUT_HTML, 'w') as f:
    f.write(html)
