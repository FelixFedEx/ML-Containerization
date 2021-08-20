from __future__ import absolute_import
import json
import pytest


@pytest.mark.integration
def test_health_endpoint(client):
    # When
    response = client.get("/ping")

    # Then
    assert response.status_code == 200
    assert response.data == b'\n'


@pytest.mark.integration
def test_invocations_endpoint(client, test_input_data):
    # When
    response = client.post("/invocations", json=test_input_data)

    # Then
    assert response.status_code == 200
    assert json.loads(response.data)


@pytest.mark.integration
def test_invocations_endpoint_for_invalid_input(client, test_invalid_input_data):
    # When
    response = client.post("/invocations", json=test_invalid_input_data)

    # Then
    assert response.status_code == 200
    assert json.loads(response.data)
