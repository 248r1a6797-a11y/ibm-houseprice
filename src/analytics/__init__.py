"""Analytics, descriptive statistics, segmentation, and deterministic insights package."""
from .descriptive import compute_market_kpis, compute_price_distribution
from .segmentation import segment_by_neighborhood, segment_by_price_tier, segment_by_size
from .insights import generate_business_insights

__all__ = [
    "compute_market_kpis",
    "compute_price_distribution",
    "segment_by_neighborhood",
    "segment_by_price_tier",
    "segment_by_size",
    "generate_business_insights",
]
