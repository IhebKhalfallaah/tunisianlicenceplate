from logging import INFO, basicConfig

from flask import Flask
from flask_migrate import Migrate

from models.license_plate import db
from routes.license_plate_bp import license_plate_bp

basicConfig(level=INFO)

app = Flask(__name__, static_url_path="/static")
app.config.from_object("config")

db.init_app(app)
migrate = Migrate(app, db)

app.register_blueprint(license_plate_bp, url_prefix="/")

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0")
