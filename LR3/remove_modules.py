"""
Скрипт для удаления папки modules.
Вспомогательный скрипт для очистки проекта от временных файлов.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import os

if os.path.exists('modules'):
    try:
        shutil.rmtree('modules')
        print('✓ Папка modules успешно удалена')
    except Exception as e:
        print(f'✗ Ошибка при удалении: {e}')
else:
    print('Папка modules не найдена')

