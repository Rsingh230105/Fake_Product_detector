# Fake Product Detector

AI-powered web application that detects counterfeit food products using deep learning, OCR, and barcode verification — containerized with Docker and deployed on AWS via Terraform (Infrastructure as Code) with a fully automated CI/CD pipeline.

## Overview

The system analyzes a product image and returns a **Real / Fake** verdict with a confidence score, by combining four independent signals:

- **MobileNetV2 CNN** — image classification (transfer learning)
- **Tesseract OCR** — extracts label text (FSSAI license, expiry date, batch number)
- **OpenCV** — image preprocessing and packaging analysis
- **pyzbar** — barcode detection and scoring

## Architecture

```
                              ┌─────────────────────────┐
                              │        GitHub            │
                              │  push to "prod" branch   │
                              └────────────┬─────────────┘
                                           │
                                    GitHub Actions
                          ┌────────────────┼────────────────┐
                          │                │                │
                   Build Docker      Push to Docker      Deploy via
                    image (multi-        Hub (private        AWS SSM
                    stage build)          repo)          (no SSH port)
                          │                │                │
                          └────────────────┴────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────┐
                    │              AWS (Terraform)              │
                    │                                            │
                    │   EC2 (Docker) ───────────► RDS PostgreSQL │
                    │        │                     (private VPC) │
                    │        ▼                                   │
                    │       S3 (media + ML model storage)        │
                    │                                            │
                    │   IAM Roles — least-privilege, no static   │
                    │   AWS keys on the instance                 │
                    └──────────────────────────────────────────┘
                                           │
                                           ▼
                                  Public IP:8000
                          (accessible from anywhere)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 4.2, Django REST Framework |
| ML / CV | TensorFlow/Keras (MobileNetV2), OpenCV, Tesseract OCR, pyzbar |
| Database | PostgreSQL (AWS RDS, private subnet, SSL required) |
| Storage | AWS S3 (media uploads + ML model, private, IAM-role access) |
| Static files | WhiteNoise |
| Containerization | Docker (multi-stage build), Docker Compose |
| Infrastructure | Terraform (VPC, EC2, RDS, S3, IAM, Security Groups) |
| CI/CD | GitHub Actions → Docker Hub → AWS SSM |
| Production server | Gunicorn |

## Deployment Pipeline

Every push to the `prod` branch triggers a fully automated pipeline:

1. **Build** — Docker image built from a multi-stage Dockerfile (compiles dependencies in a discarded `builder` stage; final image ships only runtime artifacts)
2. **Push** — image pushed to a private Docker Hub repository, tagged with both `multistage` (rolling) and the Git commit SHA (immutable, for rollback)
3. **Deploy** — GitHub Actions authenticates as a dedicated, least-privilege IAM user and sends a deployment command to the EC2 instance via **AWS Systems Manager (SSM)** — no SSH port is ever opened for CI/CD, and no private key is stored in GitHub

The EC2 instance itself is fully self-provisioning: on boot, it installs Docker, generates its own `.env` (using Terraform-supplied RDS credentials, an auto-generated Django secret key, and its own detected public IP), pulls the application image, and starts the stack — so `terraform apply` alone is enough to bring the whole system up from nothing.

## Local Development

The project runs locally with Docker Compose, using a local PostgreSQL container (production uses AWS RDS instead — see below).

```bash
git clone https://github.com/Rsingh230105/Fake_Product_detector.git
cd Fake_Product_detector
cp .env.example webapp/.env   # fill in the required values
docker compose up -d
```

App available at `http://localhost:8000`.

Useful commands:

```bash
docker compose logs -f web       # follow application logs
docker compose exec web bash     # shell into the running container
docker compose down              # stop and remove containers (data volume persists)
```

### Without Docker (manual setup)

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux/Mac

pip install -r requirements.txt
# Install Tesseract OCR separately (see https://github.com/UB-Mannheim/tesseract/wiki on Windows,
# or `sudo apt-get install tesseract-ocr` on Linux)

cd webapp
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## Infrastructure (Terraform)

All AWS infrastructure is defined as code:

```bash
cd terraform/
terraform init
terraform plan
terraform apply
```

This provisions:
- A VPC with public and private subnets across two Availability Zones
- An EC2 instance (public subnet) running the Docker-based application
- An RDS PostgreSQL instance (private subnet, not publicly accessible)
- An S3 bucket for media and ML model storage
- IAM roles for the EC2 instance (S3 + SSM access) and a separate, scoped-down IAM user for CI/CD
- Security groups following least-privilege: SSH restricted to a single admin IP, application port open publicly, database reachable only from the application's security group

## Security

- **Container**: runs as a non-root user; multi-stage build keeps compilers and build tools out of the production image
- **Secrets**: never committed to the repo or baked into the image — injected via environment variables, generated per-deployment where possible (Django secret key), or held in GitHub Actions secrets
- **Database**: RDS is not publicly accessible; connections require SSL
- **CI/CD access**: AWS SSM instead of SSH — the deployment port (22) stays closed to the internet; the CI/CD IAM user can only send SSM commands and describe instances, nothing else
- **Dependency scanning**: images are scanned with Docker Scout; known-fixable CVEs (including a Django SQL-injection advisory) are patched promptly
- **File upload validation**: magic-byte verification, size limits, MIME-type whitelist

## Project Structure

```
Fake_Product_detector/
├── webapp/
│   ├── ai_product_verification_system/   # Django settings/urls/wsgi
│   ├── detector/                         # Main app: models, views, validators
│   │   └── utils/ml_utils.py             # ML inference (MobileNetV2, OCR, barcode)
│   ├── templates/
│   └── static/
├── Dockerfile                            # Multi-stage build
├── entrypoint.sh                         # Model download, migrations, Gunicorn start
├── docker-compose.yml
├── terraform/                            # Infrastructure as Code
├── .github/workflows/deploy.yml          # CI/CD pipeline
├── requirements.txt
└── .env.example
```

## API Endpoints

```
POST /api/detect/            Upload an image for analysis
                              → { prediction, confidence, detailed_report }

GET  /dashboard/             User's analysis history

GET  /admin-report/<id>/     Detailed admin analysis report
```

## Model Training

The MobileNetV2 model is trained separately (see `model_pipeline/`) and not stored in this repository — it is pulled from S3 at container startup.

- Dataset: minimum 1,000 images per class (Real / Fake), multiple angles (front, back, side, barcode)
- Input size: 224×224×3
- Optimizer: Adam · Loss: Binary crossentropy

## Roadmap

- [ ] Elastic IP for a stable public address
- [ ] Custom domain + HTTPS (Route 53 + ACM)
- [ ] CloudWatch monitoring and alerting
- [ ] Auto Scaling Group for horizontal scaling
- [ ] Government FSSAI database integration

## License

For educational and research purposes.

---

**Note:** This is a portfolio/learning project demonstrating a full containerization and cloud-deployment workflow (Docker → Terraform → CI/CD → AWS). For production use with real consumer data, integrate with official verification databases and obtain the necessary regulatory certifications.