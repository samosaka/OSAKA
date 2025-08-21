from flask import Flask  
from flask_cors import CORS
import os

def create_app():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    FRONTEND_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../frontend'))

    app = Flask(
        __name__,
        static_folder=FRONTEND_PATH,  # Serve /js, /css, /assets
        static_url_path=''
    )
    
    CORS(app)

    # Register Blueprints
    from app.routes.base import base_bp
    from app.routes.meta import meta_bp
    from app.routes.history import history_bp
    from app.routes.algorithm import algorithm_bp
    from app.routes.agent import agent_bp
    from app.routes.backtest import backtest_bp
    from app.routes.analyze import analyze_bp

    app.register_blueprint(base_bp)
    app.register_blueprint(meta_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(algorithm_bp, url_prefix="/api")
    app.register_blueprint(agent_bp, url_prefix="/api")
    app.register_blueprint(backtest_bp, url_prefix="/api")
    app.register_blueprint(analyze_bp, url_prefix="/api")

    return app
