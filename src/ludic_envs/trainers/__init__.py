# Modal app exports
try:
    from .modal_training_app import app as training_app
except ImportError:
    # Modal may not be installed
    training_app = None