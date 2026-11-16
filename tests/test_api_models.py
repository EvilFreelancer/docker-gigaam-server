from unittest.mock import patch

from tests._support import ApiTestCase


class ApiModelTests(ApiTestCase):
    def test_models_list_matches_openai_model_list_shape(self):
        with patch.object(self.api, "engine", self.engine):
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        model = response.json()["data"][0]
        self.assertEqual(model["object"], "model")
        self.assertEqual(model["id"], self.engine.models[0])
        self.assertIn("created", model)
        self.assertIn("owned_by", model)

    def test_retrieve_model_returns_model_card(self):
        with patch.object(self.api, "engine", self.engine):
            response = self.client.get("/v1/models/gigaam-v3-e2e-rnnt")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], self.engine.models[0])

    def test_unknown_model_uses_openai_error_envelope(self):
        with patch.object(self.api, "engine", self.engine):
            response = self.client.get("/v1/models/not-a-model")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["param"], "model")
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_models_requires_configured_bearer_key(self):
        with patch.object(self.api, "API_KEY", "secret"), patch.object(self.api, "engine", self.engine):
            denied = self.client.get("/v1/models")
            allowed = self.client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(denied.status_code, 401)
        self.assertIn("error", denied.json())
        self.assertEqual(allowed.status_code, 200)

    def test_models_reports_loading_as_openai_error(self):
        with patch.object(self.api, "engine", None):
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["type"], "server_error")

    def test_health_reflects_loaded_engine(self):
        with patch.object(self.api, "engine", self.engine):
            response = self.client.get("/health")
        self.assertEqual(response.json(), {
            "status": "ok", "models": self.engine.models, "device": self.api.DEVICE
        })
