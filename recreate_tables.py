import logging
from app import db, app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_tables():
    """Recreate all database tables - CAUTION: This will delete all existing data"""
    logger.info("Recreating database tables...")
    
    with app.app_context():
        # Import models to ensure they are registered with SQLAlchemy
        import models
        
        # Drop all tables
        logger.info("Dropping all tables...")
        db.drop_all()
        
        # Create all tables
        logger.info("Creating all tables...")
        db.create_all()
        
        logger.info("Database tables recreated successfully.")

if __name__ == "__main__":
    recreate_tables()