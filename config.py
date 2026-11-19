"""Flask configuration, read from environment variables (see .env.example)."""
import os

SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32)

DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

SQLALCHEMY_TRACK_MODIFICATIONS = False

if os.environ.get("BACKEND_ENV") == "production":
    _default_db = "mysql+mysqlconnector://back:password@db/license_plate"
else:
    _default_db = "mysql+mysqlconnector://root:password@localhost/license_plate"

SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", _default_db)
