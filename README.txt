1. Title
--------
Cross-modal Plant Disease Detection System based on CLIP Architecture

2. Description
--------------
This project is a deep learning framework designed for plant disease 
identification. It utilizes the CLIP (Contrastive Language-Image 
Pre-training) architecture, combining a Vision Transformer (ViT-B/16) with 
a Text Transformer. By fusing image features with textual descriptions, 
the system achieves high-precision classification of plant diseases. 
The framework includes a specialized Transformer-based Fusion Module 
and advanced data augmentation strategies.

3. Dataset Information
----------------------
The project supports multiple dataset formats, primarily:
* PlantDoc: Includes images and corresponding JSON-formatted annotations.
* Zhiwubindu: Covers various plants (Apple, Corn, Tomato, Potato, etc.) 
  and their specific diseases (Scab, Rust, Early/Late Blight, Mosaic 
  Virus, etc.).
* Structure: Annotations are typically stored in the 'captions' directory, 
  and images are stored in the 'images' directory.

4. Code Information
-------------------
* main.py: The main entry point for the project. Handles training loops, 
  validation, logging, and model saving.
* model/fusion_model.py: Implements the Fusion module using a Transformer 
  Decoder for cross-modal interaction.
* model/tbps_model.py: Core implementation of the CLIP architecture 
  including visual and textual branches.
* options.py: Handles command-line argument parsing.
* config/config.yaml: Central configuration file defining hyperparameters, 
  model paths, and device settings.
* shell/train.sh: Shell script for launching distributed training.

5. Usage Instructions
---------------------
1. Environment Setup: Ensure all dependencies in requirements.txt are 
   installed.
2. Configuration: Modify 'image_dir' and 'anno_dir' in config/config.yaml 
   to point to your local dataset paths.
3. Single-GPU Training: Run 'python main.py'.
4. Distributed Training: Run 'sh shell/train.sh'.
5. Evaluation: The system automatically evaluates the model during 
   training, reporting Accuracy, Precision, Recall, and F1-score. 
   Loss and Accuracy curves are generated upon completion.

6. Requirements
---------------
Key dependencies include (see requirements.txt for full list):
* torch == 1.13.0
* torchvision == 0.14.0
* timm == 0.6.11
* wandb, pyyaml, easydict, matplotlib, regex, ftfy

7. Methodology
--------------
1. Backbone: Uses a pre-trained ViT-B-16 CLIP model.
2. Feature Fusion: Employs a TransformerDecoderLayer to perform 
   attention-based interaction between Query Embeddings and visual/text 
   features.
3. Training Strategy:
   - Cosine annealing scheduler for learning rate management.
   - Automatic Mixed Precision (AMP) for computational efficiency.
   - Multi-component loss function integrating Cross-Entropy and 
     various contrastive learning ratios (CITC, RITC, etc.).

8. License & Contribution
-------------------------
* License: Please follow the open-source license agreement of the original 
  author or institution.
* Contribution: Contributions to improve the disease recognition 
  algorithms or bug reports are welcome via Issues or Pull Requests.

