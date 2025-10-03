from dataclasses import dataclass
from typing import Dict


@dataclass
class APIConfig:
    """Configuration for API endpoints"""

    name: str
    base_url: str
    headers: Dict[str, str] = None
    auth_token: str = None
    endpoint: str = "/predict"
    method: str = "POST"
    request_format: str = "json"  # json, form, query
    response_format: str = "json"  # json, text
    max_requests_per_second: float = 10.0
    timeout: float = 30.0
    custom_request_formatter: callable = None  # Custom function to format requests
    custom_response_parser: callable = None  # Custom function to parse responses
