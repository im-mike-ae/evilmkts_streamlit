# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Winter Coalition Market Stats Viewer - A Streamlit web application for EVE Online market analysis providing real-time market data visualization, doctrine analysis, and inventory management for Winter Coalition.

**Critical Architecture Note:** This is a read-only frontend application. Market data updates are handled by a separate backend repository (https://github.com/OrthelT/mkts_backend) that calls ESI APIs and updates the Turso remote database. This application only syncs from and reads Turso data.

## Development Commands

### Setup and Installation
```bash
# Install dependencies (uv is the preferred package manager)
uv sync

# Run the application
uv run streamlit run app.py

# Development mode with hot reload
uv run streamlit run app.py --server.runOnSave true
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_get_market_history.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=term-missing

# Run specific test markers
uv run pytest -m unit -v
uv run pytest -m "not slow" -v
```

### Database Operations
```bash
# Sync database from remote
uv run python -c "from config import DatabaseConfig; db = DatabaseConfig('wcmkt'); db.sync()"

# Check database integrity
uv run python -c "from config import DatabaseConfig; db = DatabaseConfig('wcmkt'); print(db.integrity_check())"
```

### Code Quality
```bash
# Lint with Ruff
uv run ruff check .

# Auto-format with Ruff
uv run ruff format .
```

## High-Level Architecture

### Database Architecture: Turso Embedded Replica Pattern

The application uses a sophisticated multi-database architecture with Turso's embedded-replica feature:

**Three Databases:**
- `wcmktprod.db` - Market orders, statistics, doctrines, fits (synced from Turso)
- `sdelite2.db` - EVE Online Static Data Export (items, groups, categories)
- `buildcost.db` - Manufacturing structures, industry indices, rigs

**DatabaseConfig Class (`config.py`):**
- Centralized database connection manager with RWLock concurrency control
- Properties: `mkt_engine`, `sde_engine`, `bc_engine` for SQLAlchemy access
- Methods: `sync()`, `integrity_check()`, `validate_sync()`
- Automatic malformed database detection and recovery via remote fallback

**RWLock Concurrency Pattern:**
The `DatabaseConfig` implements a custom read-write lock allowing:
- Multiple concurrent readers (queries don't block each other)
- Exclusive writer access (sync operations block all reads/writes)
- Thread-safe database access across Streamlit's multi-threaded environment
- Implementation in `config.py` lines 21-83

**Data Flow:**
1. Backend repo → ESI API → Turso remote database (market data updates)
2. Frontend `sync()` → Turso → Local SQLite files (via libsql embedded replica)
3. Streamlit pages → Local SQLite queries (fast reads with RWLock protection)

**Malformed Database Recovery (`db_handler.py`):**
- `read_df()` function auto-detects database corruption errors
- Attempts automatic `sync()` to repair local database
- Falls back to remote Turso queries if local repair fails
- Transparent to calling code - applications don't need error handling

### Core Module Responsibilities

**`config.py` - DatabaseConfig:**
- Database connection management (SQLAlchemy engines)
- RWLock implementation for concurrent access
- Sync operations with integrity validation
- Remote fallback handling

**`db_handler.py` - Data Access Layer:**
- `read_df()` - Read queries with malformed DB recovery
- `new_read_df()` - Alternative read method with RWLock context managers
- Cached data fetchers: `get_all_mkt_stats()`, `get_all_mkt_orders()`, `get_all_market_history()`
- All use `@st.cache_data` with 15-minute TTL for performance

**`models.py` & `sdemodels.py` - ORM Layer:**
- SQLAlchemy models using modern `mapped_column()` syntax
- Market models: MarketStats, MarketOrders, MarketHistory
- Doctrine models: Doctrines, DoctrineFits, ShipTargets
- SDE models: InvTypes, InvGroups, InvCategories

**`doctrines.py` - Business Logic:**
- Doctrine fit management and aggregation
- Cost calculations (hull + modules)
- Target inventory level handling
- Fit data merging with market availability

**`utils.py` - API Integration:**
- `fetch_industry_indices()` - ESI API calls with rate limiting
- `lookup_jita_prices()` - Jita price lookups with Fuzzworks fallback
- Proper user agent headers for EVE API compliance

**`market_metrics.py` - UI Components:**
- Market ISK volume charting (Plotly)
- Historical metrics display (30-day averages)
- Moving average calculations (3/7/14/30 day)
- Outlier handling (cap/remove/none)

### Page Structure (`pages/` directory)

All pages follow consistent patterns:
1. Import `DatabaseConfig` for database access
2. Use centralized logging from `logging_config.py`
3. Cache expensive operations with `@st.cache_data`
4. Clear cache after database modifications

**Five Main Pages:**
- `market_stats.py` - Primary market data visualization
- `doctrine_status.py` - Doctrine fit status and costs
- `doctrine_report.py` - Detailed doctrine analysis
- `low_stock.py` - Low inventory alerts
- `build_costs.py` - Manufacturing cost analysis

### Configuration Files

**`settings.toml` - Application Settings:**
- Ship role definitions (dps, logi, links, support)
- Special cases: Ship + Fit ID → Role mapping (e.g., Vulture fit 369 = DPS, fit 475 = Links)
- Outlier handling defaults
- Validated by `tests/test_settings_toml.py` suite

**`.streamlit/secrets.toml` - Credentials (git-ignored):**
```toml
[secrets]
TURSO_DATABASE_URL = "libsql://..."
TURSO_AUTH_TOKEN = "..."
SDE_URL = "libsql://..."
SDE_AUTH_TOKEN = "..."
```

## Development Guidelines

### Database Access Patterns

**Always use DatabaseConfig for database access:**
```python
from config import DatabaseConfig

db = DatabaseConfig("wcmkt")  # or "sde" or "build_cost"
engine = db.engine  # Local SQLite engine

# For queries
from db_handler import read_df
df = read_df(db, "SELECT * FROM marketstats WHERE type_id = :id", {"id": 34})

# Or use cached helpers
from db_handler import get_all_mkt_stats
df = get_all_mkt_stats()  # Auto-cached, 15 min TTL
```

**Never:**
- Directly create SQLAlchemy engines
- Hard-code database paths
- Skip malformed DB error handling (use `read_df()`)
- Write market data (this is read-only for market data)

### Concurrency and Caching

**RWLock Usage:**
The RWLock is handled automatically by `DatabaseConfig`. When implementing new database operations:
- Read operations automatically acquire read locks via `db.engine.connect()`
- Sync operations automatically acquire write locks
- Don't bypass the engine properties

**Streamlit Caching:**
- Use `@st.cache_data(ttl=900)` for expensive computations (15 minutes default)
- Use `@st.cache_resource` for database engines
- Clear caches after database sync: `st.cache_data.clear()`

### Testing Approach

The test suite focuses on public API behavior, not implementation details:

**What to Test:**
- Function return types and data structure
- Data validation and edge cases
- API contracts (function signatures)
- Configuration file structure (`settings.toml`)

**What NOT to Test:**
- Internal error handling mechanisms
- SQL query structure
- Retry logic details

**Test Markers:**
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests

**Run specific markers:**
```bash
uv run pytest -m unit
uv run pytest -m "not slow"
```

### Ship Role Configuration

Ship roles are defined in `settings.toml` under `[ship_roles]`. Four standard categories:
- `dps` - Primary damage dealers
- `logi` - Logistics/healing ships
- `links` - Command ships and fleet boosters
- `support` - EWAR, tackle, interdiction

**Special Cases:** Some ships have role determined by fit ID:
```toml
[ship_roles.special_cases.Vulture]
"369" = "DPS"    # Fit ID 369 assigns DPS role
"475" = "Links"  # Fit ID 475 assigns Links role
```

When modifying ship roles:
1. Edit `settings.toml`
2. Run `uv run pytest tests/test_settings_toml.py -v` to validate
3. No code changes needed - roles are loaded at runtime

### Logging

Use the centralized logging system:
```python
from logging_config import get_logger

logger = get_logger(__name__)
logger.info("Operation completed")
logger.error("Error details")
```

Logs are written to `logs/` directory (git-ignored) with automatic rotation.

## Common Patterns

### Adding a New Page

1. Create file in `pages/` with emoji prefix: `📊_new_page.py`
2. Import required modules:
```python
import streamlit as st
from config import DatabaseConfig
from logging_config import get_logger

logger = get_logger(__name__)
db = DatabaseConfig("wcmkt")
```
3. Register in `app.py` pages dictionary
4. Use `@st.cache_data` for expensive operations
5. Follow existing page patterns for consistency

### Querying Market Data

```python
from db_handler import read_df, get_all_mkt_stats

# Cached query (preferred for repeated queries)
df = get_all_mkt_stats()  # Returns DataFrame with all market stats

# Custom query with parameters
query = "SELECT * FROM marketstats WHERE type_id = :type_id"
df = read_df(db, query, {"type_id": 34})

# The read_df function automatically:
# - Handles malformed database errors
# - Attempts sync and repair
# - Falls back to remote on failure
```

### Database Sync Operations

```python
from config import DatabaseConfig

db = DatabaseConfig("wcmkt")

# Sync from Turso (with RWLock protection)
db.sync()

# Check integrity
is_valid = db.integrity_check()  # Returns True/False

# Validate sync success
sync_valid = db.validate_sync()  # Compares timestamps
```

## Important Constraints

### Read-Only Market Data
This application DOES NOT write market data. Market updates are handled by the separate mkts_backend repository. When working on this codebase:
- Never implement ESI API calls for market data
- Don't add write operations for marketorders/marketstats tables
- Sync operations only read from Turso, never write to it

### Package Manager
Use `uv` (not pip directly) for all dependency operations:
```bash
uv add package-name       # Add dependency
uv sync                   # Install all dependencies
uv run command            # Run command in uv environment
```

### Python Version
Requires Python 3.12+. Check `.python-version` file for exact version.

## Troubleshooting

### Database Issues
- **Malformed database**: Automatically handled by `read_df()` - syncs and falls back to remote
- **Sync failures**: Check Turso credentials in `.streamlit/secrets.toml`
- **Slow queries**: Clear Streamlit cache with `st.cache_data.clear()`
- **Connection errors**: Review logs in `logs/` directory

### Performance
- Cache expensive operations with `@st.cache_data`
- Monitor RWLock contention in logs (watch for "waiting for" messages)
- Database integrity checks log timing information

### Configuration Validation
Run `uv run pytest tests/test_settings_toml.py -v` to validate `settings.toml` structure before committing changes.
