# lnn-stock-evaluation
Liquid Neural Net that seeks to evaluate stock price performance and forecast movements, equipped with 5 layers of abstraction.


lnn-stock-evaluation/
├── README.md
├── requirements.txt
├── requirements_jetson.txt
├── config.py
├── main.py
├── setup.py
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── validator.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── liquid_network.py
│   │   ├── base_model.py
│   │   └── model_utils.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── pattern_recognition.py
│   │   ├── dimensionality_reduction.py
│   │   ├── feature_engineering.py
│   │   └── temporal_features.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── backtesting.py
│   │
│   ├── options/
│   │   ├── __init__.py
│   │   ├── scanner.py
│   │   ├── greeks.py
│   │   └── ladder_analysis.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   ├── data_utils.py
│   │   ├── system_monitor.py
│   │   └── logging_config.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── data_pipeline.py
│       ├── inference_pipeline.py
│       └── batch_processor.py
│
├── deployment/
│   ├── jetson/
│   │   ├── install_jetson.sh
│   │   ├── setup_environment.sh
│   │   ├── deploy.sh
│   │   ├── systemd/
│   │   │   └── lnn-stock.service
│   │   └── docker/
│   │       ├── Dockerfile.jetson
│   │       └── docker-compose.jetson.yml
│   │
│   └── local/
│       ├── install_local.sh
│       └── setup_dev.sh
│
├── notebooks/
│   ├── exploration/
│   │   ├── data_exploration.ipynb
│   │   └── model_experimentation.ipynb
│   │
│   └── analysis/
│       ├── performance_analysis.ipynb
│       └── feature_importance.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   │   └── test_loader.py
│   ├── test_models/
│   │   └── test_liquid_network.py
│   └── test_features/
│       └── test_feature_engineering.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── results/
│
├── configs/
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── jetson_config.yaml
│
├── scripts/
│   ├── download_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_inference.py
│
└── docs/
    ├── installation.md
    ├── jetson_setup.md
    ├── api_reference.md
    └── troubleshooting.md
