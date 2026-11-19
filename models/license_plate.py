from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class LicensePlate(db.Model):
    __tablename__ = 'licence_plate'
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(14))
    paths=db.Column(db.String(300))

    @property
    def serialize(self):
        return {
            'id': self.id,
            'license_plate': self.license_plate,
            'paths': self.paths,
        }