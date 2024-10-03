import logging

def setup_logging(level=logging.DEBUG):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicate logs in Jupyter
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler that writes log messages to the console (Jupyter cell output)
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Optionally, adjust logging levels for external libraries
    # logging.getLogger('some_library').setLevel(logging.WARNING)

    #print("Logging has been configured.")