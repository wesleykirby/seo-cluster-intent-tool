# SEO Cluster & Intent Tool

## Overview
This project is a Streamlit web application that provides keyword clustering and intent classification functionality for SEO purposes. It includes an active learning workflow for collecting human feedback on low-confidence predictions.

## Project Structure
- `app.py` - Main Streamlit application interface
- `cluster/` - Core clustering and intent analysis modules
  - `pipeline.py` - Main clustering pipeline and intent classification
  - `active_learning.py` - Queue management for human labeling
  - `models/intent/` - Intent classification model components
- `data/` - Data storage directory
  - `training_data.csv` - Training examples
  - `label_queue.csv` - Queue for human labeling (created during usage)
- `scripts/` - Utility scripts
  - `weekly_retrain.py` - Weekly retraining workflow

## Setup Completed
- Python 3.11 installed
- All required packages installed from requirements.txt
- Streamlit workflow configured to run on port 5000
- Data directories created
- Deployment configuration set for autoscale deployment
- Application successfully running and tested

## Current State
- Application is fully functional and running
- Ready for production deployment
- Web interface accessible at the configured port
- All core features working: file upload, clustering, intent classification, active learning queue

## Recent Changes (September 16, 2025)
- Initial project import and setup completed
- Environment configured for Replit hosting
- All dependencies installed and verified working
- Deployment configuration added for production publishing

## User Preferences
- No specific preferences recorded yet

## Project Architecture
- Frontend: Streamlit web application
- Backend: Python-based clustering and ML pipeline
- Data Storage: CSV files for training data and label queue
- ML Components: TF-IDF vectorization with cosine similarity for clustering, intent classification models
- Deployment: Configured for autoscale deployment suitable for web applications