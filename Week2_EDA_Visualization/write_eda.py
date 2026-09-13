
content = open('week2_eda_source.txt', 'r', encoding='utf-8').read()
with open('week2_eda.py', 'w', encoding='utf-8') as f:
    f.write(content)
import os
print('Written', os.path.getsize('week2_eda.py'), 'bytes')
