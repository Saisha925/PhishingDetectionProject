# app/__init__.py

from flask import Flask

def create_app():
    app = Flask(__name__)

    # Configuration and initialization
    # e.g., app.config.from_object('config.Config')

    with app.app_context():
        # Register blueprints
        from .routes import main
        app.register_blueprint(main)

    return app
