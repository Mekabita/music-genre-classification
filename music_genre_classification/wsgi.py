"""
WSGI config for music_genre_classification project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

setting_module = 'music_genre_classification.deployment' if 'WEBSITE_HOSTNAME' in os.environ else 'music_genre_classification.settings'

os.environ.setdefault("DJANGO_SETTINGS_MODULE", setting_module)

application = get_wsgi_application()
