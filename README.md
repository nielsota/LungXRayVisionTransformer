# LungXRayVisionTransformer

This repository contains code aimed at deploying a Vision Transformer trained to classify diseases based on lung X-Rays. More importantly however, it server as a blueprint for future ML-deployment projects. Data is downloaded and preprocessed in notebooks, that run both locally and on SageMaker. Using a .yaml pipeline, data and models can be retrieved from S3. The serve_pytorch contains the model, and model training script. 

