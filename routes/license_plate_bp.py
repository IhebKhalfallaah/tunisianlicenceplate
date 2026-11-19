from flask import Blueprint

from controllers.license_plate_controller import about, demo, detect, detect_multiple, index, read

license_plate_bp = Blueprint("license_plate_bp", __name__)

# Pages
license_plate_bp.route("/", methods=["GET"])(index)
license_plate_bp.route("/demo", methods=["GET"])(demo)
license_plate_bp.route("/about", methods=["GET"])(about)

# API
license_plate_bp.route("/detect", methods=["POST"])(detect)
license_plate_bp.route("/detect/all", methods=["POST"])(detect_multiple)
license_plate_bp.route("/read", methods=["POST"])(read)
