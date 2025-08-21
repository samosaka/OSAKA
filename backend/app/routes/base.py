from flask import Blueprint, send_from_directory, jsonify
import os

base_bp = Blueprint('base', __name__)

@base_bp.route('/')
def serve_index():
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../frontend'))
    return send_from_directory(frontend_path, 'index.html')