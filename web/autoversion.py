import random
import sys
from os import path

ver_base = 10**15
ver_num = str(random.randint(ver_base, 10 * ver_base - 1))

BUILD_DIR = sys.argv[1]

INDEX_HTML = path.join(BUILD_DIR, "index.html")
with open(INDEX_HTML, 'r') as f:
    html = f.read()
    html = html.replace('{{ver}}', ver_num)
with open(INDEX_HTML, 'w') as f:
    f.write(html)

MAIN_JS = path.join(BUILD_DIR, "js", "main.js")
with open(MAIN_JS, 'r') as f:
    js = f.read()
    js = js.replace("'volrend_web.wasm'", f"'volrend_web.wasm?v={ver_num}'")
with open(MAIN_JS, 'w') as f:
    f.write(js)

