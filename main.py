# Load environment variables from .env file first
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from app import app

# Import routes to register them
import routes  # noqa: F401

# Import market data routes 
from market_routes import market_bp

# Register blueprints
app.register_blueprint(market_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG", "True").lower() == "true")
