from src.config import settings


def test_settings_load():
    assert settings.APP_NAME == "Cost-Efficient RAG Service"
    assert settings.VECTOR_DB_TYPE in ["lancedb", "chromadb"]
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.TOP_K > 0
