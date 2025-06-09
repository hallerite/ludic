# Modal app exports
try:
    from .modal_inference_app import app as inference_app
except ImportError:
    # Modal may not be installed
    inference_app = None