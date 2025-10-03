import requests
from api_config import APIConfig
from typing import List, Tuple


class APIClient:
    """Generic API client for skill recommendation APIs"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        if config.headers:
            self.session.headers.update(config.headers)
        if config.auth_token:
            self.session.headers.update(
                {"Authorization": f"Bearer {config.auth_token}"}
            )

    def predict(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Make a prediction request to the API
        Returns list of (skill_name, score) tuples
        """
        url = f"{self.config.base_url.rstrip('/')}{self.config.endpoint}"

        # Use custom request formatter if available
        if self.config.custom_request_formatter:
            data, params = self.config.custom_request_formatter(query, top_k)
            if params:
                response = self.session.request(
                    self.config.method,
                    url,
                    json=data,
                    params=params,
                    timeout=self.config.timeout,
                )
            else:
                response = self.session.request(
                    self.config.method, url, json=data, timeout=self.config.timeout
                )
        else:
            # Prepare request data based on format (legacy behavior)
            if self.config.request_format == "json":
                data = {"query": query, "top_k": top_k}
                response = self.session.request(
                    self.config.method, url, json=data, timeout=self.config.timeout
                )
            elif self.config.request_format == "form":
                data = {"query": query, "top_k": top_k}
                response = self.session.request(
                    self.config.method, url, data=data, timeout=self.config.timeout
                )
            elif self.config.request_format == "query":
                params = {"query": query, "top_k": top_k}
                response = self.session.request(
                    self.config.method, url, params=params, timeout=self.config.timeout
                )

        response.raise_for_status()

        # Use custom response parser if available
        if self.config.custom_response_parser:
            return self.config.custom_response_parser(response)
        else:
            # Parse response based on format (legacy behavior)
            if self.config.response_format == "json":
                result = response.json()
                return self._parse_json_response(result)
            else:
                return self._parse_text_response(response.text)

    def _parse_json_response(self, result: dict) -> List[Tuple[str, float]]:
        """
        Parse JSON response - adapt this method based on your API response format
        Expected formats:
        - {"predictions": [{"skill": "name", "score": 0.95}, ...]}
        - {"skills": ["skill1", "skill2"], "scores": [0.95, 0.87]}
        - [{"skill": "name", "score": 0.95}, ...]
        """
        predictions = []

        # Try different common response formats
        if "predictions" in result:
            for item in result["predictions"]:
                if isinstance(item, dict) and "skill" in item and "score" in item:
                    predictions.append((item["skill"], float(item["score"])))
                elif isinstance(item, dict) and "name" in item and "score" in item:
                    predictions.append((item["name"], float(item["score"])))
        elif "skills" in result and "scores" in result:
            skills = result["skills"]
            scores = result["scores"]
            predictions = [
                (skill, float(score)) for skill, score in zip(skills, scores)
            ]
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    if "skill" in item and "score" in item:
                        predictions.append((item["skill"], float(item["score"])))
                    elif "name" in item and "score" in item:
                        predictions.append((item["name"], float(item["score"])))
        elif "results" in result:
            for item in result["results"]:
                if isinstance(item, dict) and "skill" in item and "score" in item:
                    predictions.append((item["skill"], float(item["score"])))

        return predictions

    def _parse_text_response(self, text: str) -> List[Tuple[str, float]]:
        """Parse text response - implement based on your API text format"""
        # Example implementation for tab-separated or comma-separated format
        predictions = []
        lines = text.strip().split("\n")
        for line in lines:
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                continue

            if len(parts) >= 2:
                try:
                    skill = parts[0].strip()
                    score = float(parts[1].strip())
                    predictions.append((skill, score))
                except ValueError:
                    continue

        return predictions
