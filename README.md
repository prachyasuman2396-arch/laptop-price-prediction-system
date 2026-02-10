Laptop Price Prediction – End-to-End ML System

Deployed a full ML model with engineered features and XGBoost, served via a Flask + Gunicorn REST API with a front-end UI at:
https://laptop-price-prediction-system-f1o3.onrender.com

Overview

This project is an end-to-end machine learning system that predicts laptop prices based on hardware and design specifications.
The goal is not only to achieve low error, but to design a robust, deployable, and reproducible ML pipeline that mirrors real-world production constraints.
The system covers:
Data preprocessing & feature engineering
Multiple model experimentation
Error analysis & model comparison
Deployment as a REST API with a user-facing UI
The final model is deployed on Render and serves real-time predictions.


Problem Statement

Laptop prices are influenced by multiple interacting factors such as:
CPU & GPU class
RAM and storage configuration
Display quality
Brand positioning
The challenge is capturing non-linear interactions between these components while maintaining generalization and deployment safety.


Dataset

The dataset contains laptop specifications including:
Brand and form factor
CPU and GPU information
RAM and storage details
Screen size and display properties
Operating system
Target Variable
Price (INR)
Log-transformed during training to stabilize variance and reduce skew.


Feature Engineering

Instead of relying on raw attributes alone, the model uses engineered features to better capture real-world pricing logic:
Key Engineered Features
Feature	Reason
RamGB	Normalized memory representation
TotalStorage_GB	Combined SSD + HDD capacity
HasSSD	SSD presence impacts perceived value
HasDedicatedGPU	Strong pricing signal
GpuClass	Integrated vs Dedicated
CpuTier	Entry / Mid / High performance CPUs
CpuPowerClass	Power vs efficiency CPUs
PPI	Screen sharpness proxy
StorageType	HDD / SSD / Hybrid
IsRazer	Brand premium flag
These features significantly improved model performance compared to raw inputs.


Models Evaluated

Multiple regression models were evaluated to understand bias-variance trade-offs.


Linear Regression

Why tried:
Baseline model for interpretability.
Failure mode:
Unable to model non-linear interactions
High RMSE due to underfitting
Conclusion:
Rejected due to poor expressive power.


Random Forest Regressor

Why tried:
Ensemble to reduce variance of trees.
Strengths:
Strong improvement in RMSE
Robust to outliers
Limitations:
Larger model size
Less efficient inference
Slightly worse RMSE than XGBoost
Conclusion:
Strong candidate, but not optimal.


XGBoost Regressor (Final Model)

Why chosen:
Best RMSE on validation set
Handles non-linear feature interactions efficiently
Built-in regularization prevents overfitting
Production-ready and widely used in industry
Key advantages:
Gradient boosting captures complex pricing patterns
Efficient inference suitable for deployment
Better bias-variance balance than Random Forest


Final Choice: XGBoost achieved the lowest RMSE while maintaining stability and scalability.


Evaluation Strategy

Train/validation split
RMSE used as primary metric
Log-price predictions inverted during inference to return actual INR values
This ensured:
Robust comparison across models
Realistic price outputs for users


Deployment Architecture

The trained model is deployed as a Flask REST API and served using Gunicorn for production stability.
Components
Flask – API layer
Gunicorn – Production WSGI server
Render – Cloud hosting
Pickled model – Serialized trained model
API Endpoints
Endpoint	Description
/	UI for user input
/predict	JSON-based prediction
/health	Health check


Prediction Flow

User enters laptop specifications via UI
Raw inputs are transformed into engineered features
Model predicts log(price)
Inverse log transform applied
Final price returned in INR
This avoids training–serving skew and ensures consistent predictions.


Engineering Challenges & Learnings

Training–Serving Mismatch
Initial deployment failed due to missing engineered features.
Resolution:
Aligned inference feature engineering with training logic
Reinforced importance of end-to-end pipelines
Model Selection
Lower RMSE alone was not enough — inference speed and stability were considered.


Why This Project Matters

This project demonstrates:
Practical ML problem solving
Model failure analysis
Feature engineering intuition
Deployment readiness
Production debugging experience
It reflects real ML engineering, not just notebook experimentation.


Tech Stack

Python
Pandas, NumPy
Scikit-learn
XGBoost
Flask
Gunicorn
Render


Future Improvements

Refactor feature engineering into a custom sklearn transformer
Save a unified pipeline to eliminate duplication
Add automated input validation
Add CI checks for deployment safety


Author
Prachya Das
B.Tech (CSE – AI & ML)
Aspiring ML Engineer | End-to-End ML Systems
