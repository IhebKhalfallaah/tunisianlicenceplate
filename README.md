# 🇹🇳 Tunisian Licence Plate Recognition

> Detect, read and anonymise **Tunisian licence plates** from car photos — a Flask web app powered by ONNX deep-learning models and EasyOCR.

![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.2-000000?logo=flask&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-inference-005CED?logo=onnx&logoColor=white)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Tunisian plates have a specific layout: a serial number on the left, the word **تونس** ("Tunis") in the middle, and a registration number on the right (e.g. `202 تونس 2806`). This project detects the plate, locates the **تونس** band to split the plate in two, reads each side with OCR, blurs the plate on the original photos and stores the result in a database.

---

## 📸 Screenshots

> _Screenshots coming soon._ Add images to `docs/screenshots/` and uncomment the lines below.

<!--
| Home | Demo |
|------|------|
| ![Home](docs/screenshots/home.png) | ![Demo](docs/screenshots/demo.png) |
-->

## ✨ Features

- **Plate detection** – a custom grid-based CNN (ONNX, 256×256 input) finds the licence plate in each photo.
- **"تونس" detection** – a YOLOv5 model (ONNX, 640×640 input) locates the Arabic word *Tunis* on the cropped plate.
- **OCR** – the plate is split around the تونس band and both numbers are read with EasyOCR → `"<left> TN <right>"`.
- **Multi-image input** – send several photos of the same car; the clearest plate (widest aspect ratio) is used for reading.
- **Privacy by default** – plates are Gaussian-blurred before the images are saved.
- **Persistence** – results are stored in MySQL (SQLAlchemy + Alembic migrations).
- **Web UI + REST API** – a demo page to upload photos, plus JSON endpoints.
- **Dockerised** – one command spins up the app and its database.

## 🧠 How it works

```
 photos ──► Plate detector (model.onnx) ──► crop plate
                                              │
                                              ▼
                               "تونس" detector (tunis.onnx)
                                              │
                                 split left │ right
                                              ▼
                                   EasyOCR  ──►  "202 TN 2806"
                                              │
                 blur plates ──► save to images/ ──► MySQL record
```

## 📋 Prerequisites

- **Docker** and **Docker Compose** (recommended), _or_
- **Python 3.10** and a running **MySQL 8** server for a local setup
- ~4 GB of free disk space (EasyOCR pulls in PyTorch; OCR weights are downloaded on first run)

## 🚀 Installation

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/iheb-kh/tunisianlicenceplate.git
cd tunisianlicenceplate
docker compose up --build
```

The app is available at **http://localhost:5000** and MySQL at `localhost:3306`.
The database schema is created automatically from `database/schema.sql`.

### Option 2 — Local Python environment

```bash
git clone https://github.com/iheb-kh/tunisianlicenceplate.git
cd tunisianlicenceplate

python3.10 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then edit DATABASE_URL
export $(grep -v '^#' .env | xargs) # load the variables (or set them manually)

mysql -u root -p -e "CREATE DATABASE license_plate"
flask --app app db upgrade          # create the tables

python app.py
```

### Configuration

| Variable       | Default                                                          | Description                                  |
|----------------|------------------------------------------------------------------|----------------------------------------------|
| `DATABASE_URL` | `mysql+mysqlconnector://root:password@localhost/license_plate`   | SQLAlchemy connection string                 |
| `BACKEND_ENV`  | _(empty)_ — `production` in Docker                               | `production` uses the Docker `db` host       |
| `SECRET_KEY`   | random per start                                                 | Flask secret key                             |
| `FLASK_DEBUG`  | `1`                                                              | Flask debug mode                             |

## 🛠️ Usage

### Web interface

| Page     | URL                          |
|----------|------------------------------|
| Home     | http://localhost:5000/       |
| Demo     | http://localhost:5000/demo   |
| About    | http://localhost:5000/about  |

Open **/demo**, upload one or more photos of the same car, and the recognised plate number is displayed.

### REST API

**Read a plate** — `POST /read` (multipart, files named `image0`, `image1`, …)

```bash
curl -X POST http://localhost:5000/read \
     -F image0=@car_front.jpg \
     -F image1=@car_side.jpg
```

```json
{ "id": 1, "license_plate": "202 TN 2806" }
```

On failure: `{ "error": "error reading license plate number [detecting Tunis]" }`

**Detect the plate in one image** — `POST /detect` (file `image`)

```bash
curl -X POST http://localhost:5000/detect -F image=@car.jpg
```

```json
{ "bboxes": [412, 530, 220, 58] }
```

**Detect plates in several images** — `POST /detect/all` (files `image0`, `image1`, …)

```bash
curl -X POST http://localhost:5000/detect/all -F image0=@car.jpg
```

```json
{ "bboxes": [[[412, 530, 220, 58]]] }
```

Boxes are `[x, y, width, height]` in pixels of the original image, one list per uploaded image.

## 📁 Project structure

```
tunisianlicenceplate/
├── app.py                     # Flask entry point
├── config.py                  # Configuration (env variables)
├── controllers/               # Request handlers
├── routes/                    # URL → controller mapping (blueprint)
├── services/
│   ├── license_plate_service.py  # Detection, OCR, blurring, persistence
│   └── utils.py                  # Pure bbox / NMS / image helpers
├── models/                    # SQLAlchemy models
├── migrations/                # Alembic database migrations
├── weights/                   # ONNX models (plate detector, تونس detector)
├── templates/                 # Jinja2 HTML pages
├── static/                    # CSS, JS, images (HTML Codex template)
├── images/                    # Blurred uploads saved at runtime (git-ignored)
├── database/                  # MySQL Docker image + schema
├── notebooks/                 # Training & experimentation notebooks (Colab)
├── tests/                     # Unit tests (pytest)
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## 📓 Notebooks & dataset

The `notebooks/` folder documents how the **تونس** detector was built (Google Colab):

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Explore the annotated plate images |
| `02_yolo_dataset_preparation.ipynb` | Convert annotations to YOLO format |
| `03_yolov5_tunis_training.ipynb` | Train YOLOv5 to detect "تونس" |
| `04_training_results.ipynb` | Plot training metrics (`data/results.csv`) |

The annotation file is in `notebooks/data/`. The image dataset (567 plate images) is not included in the repository. **Dataset download: _link coming soon_.**

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests cover the pure geometry helpers (scaling, IoU, NMS, detection grid, blurring) and don't need the models or a database.

## ⚠️ Known limitations

- Each photo must contain **exactly one** detectable plate; otherwise the request fails.
- Reading relies on finding the **تونس** band, so plates in other formats (e.g. `RS`, diplomatic, motorcycles) aren't supported.
- `/detect/all` normalises pixels with `/256` while `/read` uses `/255`.
- `onnxruntime` and `Pillow` versions aren't pinned.
- `static/img/favicon.ico` is referenced by the templates but not shipped.
- The models were trained on a small dataset, so accuracy drops on low-light, angled or dirty plates.

## 🗺️ Roadmap

- [ ] Publish the training dataset
- [ ] Unify image normalisation between `/detect/all` and `/read`
- [ ] Support multiple plates per image
- [ ] Return the blurred images through the API
- [ ] Add integration tests with sample images
- [ ] CI with GitHub Actions

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

This project is released under the [MIT License](LICENSE).

The web front-end is based on the free **AI.Tech** template by [HTML Codex](https://htmlcodex.com), licensed under [CC BY 4.0](static/LICENSE.txt).
