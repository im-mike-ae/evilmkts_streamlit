"""
Create database schemas in Turso remote databases.

This script creates all necessary tables in the three Turso databases:
- wcmktprod: Market data tables (marketstats, marketorders, etc.)
- sdelite2: EVE Online Static Data Export tables
- buildcost: Manufacturing structure and industry index tables

Usage:
    uv run python create_schemas.py
"""

from config import DatabaseConfig
from logging_config import setup_logging
import models
import sdemodels
import build_cost_models

logger = setup_logging(__name__)

def create_wcmktprod_schema():
    """Create schema for wcmktprod database (market data)."""
    logger.info("="*100)
    logger.info("Creating wcmktprod schema...")
    logger.info("="*100)

    db = DatabaseConfig("wcmktprod")

    # Use remote engine to create tables in Turso
    engine = db.remote_engine

    # Create all tables defined in models.py
    models.Base.metadata.create_all(engine)

    # List created tables
    tables = models.Base.metadata.tables.keys()
    logger.info(f"Created {len(tables)} tables in wcmktprod:")
    for table in tables:
        logger.info(f"  ✓ {table}")

    logger.info("wcmktprod schema created successfully!")
    return True

def create_sdelite2_schema():
    """Create schema for sdelite2 database (EVE SDE)."""
    logger.info("="*100)
    logger.info("Creating sdelite2 schema...")
    logger.info("="*100)

    db = DatabaseConfig("sde")

    # Use remote engine to create tables in Turso
    engine = db.remote_engine

    # Create all tables defined in sdemodels.py
    sdemodels.Base.metadata.create_all(engine)

    # List created tables
    tables = sdemodels.Base.metadata.tables.keys()
    logger.info(f"Created {len(tables)} tables in sdelite2:")
    for table in tables:
        logger.info(f"  ✓ {table}")

    logger.info("sdelite2 schema created successfully!")
    return True

def create_buildcost_schema():
    """Create schema for buildcost database (manufacturing data)."""
    logger.info("="*100)
    logger.info("Creating buildcost schema...")
    logger.info("="*100)

    db = DatabaseConfig("build_cost")

    # Use remote engine to create tables in Turso
    engine = db.remote_engine

    # Create all tables defined in build_cost_models.py
    build_cost_models.Base.metadata.create_all(engine)

    # List created tables
    tables = build_cost_models.Base.metadata.tables.keys()
    logger.info(f"Created {len(tables)} tables in buildcost:")
    for table in tables:
        logger.info(f"  ✓ {table}")

    logger.info("buildcost schema created successfully!")
    return True

def create_wcmkttest_schema():
    """Create schema for wcmkttest database (test market data)."""
    logger.info("="*100)
    logger.info("Creating wcmkttest schema...")
    logger.info("="*100)

    db = DatabaseConfig("wcmkttest")

    # Use remote engine to create tables in Turso
    engine = db.remote_engine

    # Create all tables defined in models.py (same as wcmktprod)
    models.Base.metadata.create_all(engine)

    # List created tables
    tables = models.Base.metadata.tables.keys()
    logger.info(f"Created {len(tables)} tables in wcmkttest:")
    for table in tables:
        logger.info(f"  ✓ {table}")

    logger.info("wcmkttest schema created successfully!")
    return True

def main():
    """Create all database schemas."""
    logger.info("="*100)
    logger.info("INITIALIZING DATABASE SCHEMAS IN TURSO")
    logger.info("="*100)

    results = {}

    try:
        results['wcmktprod'] = create_wcmktprod_schema()
    except Exception as e:
        logger.error(f"Failed to create wcmktprod schema: {e}")
        results['wcmktprod'] = False

    try:
        results['wcmkttest'] = create_wcmkttest_schema()
    except Exception as e:
        logger.error(f"Failed to create wcmkttest schema: {e}")
        results['wcmkttest'] = False

    try:
        results['sdelite2'] = create_sdelite2_schema()
    except Exception as e:
        logger.error(f"Failed to create sdelite2 schema: {e}")
        results['sdelite2'] = False

    try:
        results['buildcost'] = create_buildcost_schema()
    except Exception as e:
        logger.error(f"Failed to create buildcost schema: {e}")
        results['buildcost'] = False

    # Summary
    logger.info("="*100)
    logger.info("SCHEMA CREATION SUMMARY")
    logger.info("="*100)
    for db_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{db_name}: {status}")

    all_success = all(results.values())
    if all_success:
        logger.info("="*100)
        logger.info("All schemas created successfully! 🎉")
        logger.info("="*100)
        logger.info("Next steps:")
        logger.info("1. Run the backend repo to populate market data")
        logger.info("2. Populate SDE data (EVE Online static data)")
        logger.info("3. Populate buildcost data (structures and indices)")
        logger.info("4. Run: uv run python -c 'from init_db import init_db; init_db()' to sync locally")
        logger.info("5. Run: uv run streamlit run app.py")
    else:
        logger.error("Some schemas failed to create. Check errors above.")

    return all_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
