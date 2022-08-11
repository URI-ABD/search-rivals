import logging
import os


LOG_LEVEL = getattr(logging, os.environ.get('RIVALS_LOG', 'INFO'))
