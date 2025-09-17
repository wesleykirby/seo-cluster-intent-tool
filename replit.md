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
- **MAJOR UPGRADE COMPLETED**: Transformed from character-based clustering to semantic ML analysis
- Smart keyword analyzer now outputs clean 4-column structure: Main Topic, Sub Topic, Modifier, Keyword
- Modifier column serves as key intent indicator (Login, App, Registration, Ghana, etc.)
- ML-powered brand discovery and pattern recognition works across different markets
- Application fully functional and ready for production deployment
- Web interface shows real-time analysis with semantic insights

## Recent Changes (September 17, 2025)
- **Complete system rebuild**: Replaced old clustering with semantic analysis engine
- **New architecture**: Added `cluster/semantic_analyzer.py` for ML-powered pattern recognition
- **Smart brand discovery**: System learns brands from data patterns, not hardcoded lists
- **Intent-focused design**: Modifier extraction prioritizes user intent indicators
- **Clean output structure**: 4-column format exactly matches user specifications
- **Cross-market scalability**: Designed to work with different regions and brand ecosystems
- **UI modernized**: Updated Streamlit interface to showcase new capabilities

## User Preferences
- No specific preferences recorded yet

## Project Architecture
- **Frontend**: Streamlit web application with semantic analysis showcase
- **Backend**: ML-powered semantic analysis pipeline
- **Core Engine**: `cluster/semantic_analyzer.py` - Smart pattern recognition system
- **Data Processing**: Single-column CSV input → 4-column structured output
- **ML Components**: Dynamic brand discovery, intent classification, semantic topic categorization
- **Key Innovation**: Modifier-driven intent analysis for SEO strategy insights
- **Deployment**: Configured for autoscale deployment suitable for web applications