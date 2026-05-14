"""
peccavi/constants.py
Shared constants for PECCAVI watermarking system.
"""

# Secret key for watermark seed derivation
# Change in production - this is the default used across all agents
SECRET_KEY = "AIISC-SECRET"

# Watermarking parameters
DEFAULT_THETA = 2.0
MIN_THETA = 0.1
MAX_THETA = 10.0
TOURNAMENT_K = 8
DETECTION_THRESHOLD = 0.52

# Policy learning parameters
ALPHA = 0.05  # REINFORCE learning rate
GAMMA = 0.99  # Discount factor
LAMBDA_WM = 0.6  # Weight for watermark score in reward
NU_QUALITY = 0.4  # Weight for text quality in reward

# Evaluation parameters
DEFAULT_GENERATIONS = 10
DEFAULT_N_PARAPHRASES = 5
SUCCESS_WATERMARK_RETENTION = 0.85
SUCCESS_AUC_ROC = 0.90
SUCCESS_READABILITY = 4.5
