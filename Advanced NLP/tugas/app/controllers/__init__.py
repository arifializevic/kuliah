from flask import Flask


def create_app():
    app = Flask(__name__, template_folder='../templates',
                static_folder='../static')
    # app.config.from_object('config.Config')

    # Security headers
    @app.after_request
    def after_request(response):
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    # Register blueprints
    from .index import index_bp

    app.register_blueprint(index_bp, url_prefix='/')

    return app
