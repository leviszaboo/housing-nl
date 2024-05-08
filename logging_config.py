import colorlog
import logging

def setup_logging():
    handler = colorlog.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        '%(purple)s%(asctime)s - %(log_color)s%(levelname)-8s - %(module)-8s - %(white)s%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', 
        log_colors={
            'DEBUG': 'green',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
        },
        secondary_log_colors={
            'message': {
                'DEBUG': 'blue',
                'INFO': 'white',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'purple'
            }
        },
        style='%'
    )

    # Set the formatter to the handler
    handler.setFormatter(formatter)

    # Set up the root logger
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
