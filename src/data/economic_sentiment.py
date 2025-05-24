import yaml
import os

class EconomicSentimentLayer:
    def __init__(self, config_path: str = "config/economic_config.yaml"):
        """
        Initialize with YAML configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load API keys
        api_keys_path = "config/api_keys.yaml"
        if os.path.exists(api_keys_path):
            with open(api_keys_path, 'r') as f:
                api_config = yaml.safe_load(f)
                fred_api_key = api_config['fred']['api_key']
        else:
            # Fallback to environment variable
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                raise ValueError("No FRED API key found. Set FRED_API_KEY env var or create config/api_keys.yaml")
        
        self.fred = Fred(api_key=fred_api_key)
        
        # Extract indicators from config
        self.indicators = {}
        for name, settings in self.config['economic_indicators'].items():
            if settings['enabled']:
                self.indicators[name] = settings['series_id']
        
        # Setup other parameters from config
        self.lookback_months = self.config['processing']['lookback_months']
        self.cache_dir = self.config['data_sources']['cache']['directory']
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger = self._setup_logging()
