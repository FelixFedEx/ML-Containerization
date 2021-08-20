from __future__ import absolute_import
import pytest
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


@pytest.fixture()
def client():
    from api.wsgi import app
    app.testing = True
    with app.test_client() as client:
        yield client  # Has to be yielded to access session cookies


@pytest.fixture()
def test_input_data():
    api_data = {
        "ShortDesc": "[Sansa][IEC][HW][SI2][][X4] EESQTM 28.1  Storage SD CARD test fail.",
        "LongDesc": "Failed items : Clock Rise Time: 1.152ns (spec. Max :0.96ns)  Clock Fall Time 1.373ns "
                    "(spec. Max :0.96ns)",
        "StepsToRepro": "Follow EESQTM 28.1 test procedure.",
        "Frequency": "Always: 100%",
        "BizSegId": 1
    }
    return api_data


@pytest.fixture()
def test_invalid_input_data():
    invalid_api_data = {
        "ShortDesc": "[Sansa][IEC][HW][SI2][][X4] EESQTM 28.1  Storage SD CARD test fail.",
        "LongDesc": 123,
        "StepsToRepro": "Follow EESQTM 28.1 test procedure.",
        "Frequency": "Always: 10",
        "BizSegId": "ddd1"
    }
    return invalid_api_data
