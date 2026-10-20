"""Public model imports register the complete schema without domain import cycles."""

import pytest
from sqlalchemy import inspect

from falcon_service import models
from falcon_service.db import Base, Database

PUBLIC_MODELS = {
    "BrowserSession": "browser_sessions",
    "Export": "exports",
    "FeatureArtifact": "feature_artifacts",
    "Gallery": "galleries",
    "GalleryIndexArtifact": "gallery_index_artifacts",
    "GalleryVersion": "gallery_versions",
    "ImportItem": "import_items",
    "ImportSession": "import_sessions",
    "Job": "jobs",
    "Observation": "observations",
    "ObservationDigest": "observation_digests",
    "Review": "reviews",
    "ReviewAnnotation": "review_annotations",
    "Search": "searches",
}


@pytest.fixture
def database():
    database = Database("sqlite:///:memory:")
    database.initialize()
    yield database
    database.close()


def test_public_imports_register_every_model_on_the_shared_base():
    assert set(models.__all__) == set(PUBLIC_MODELS)
    exported_models = {getattr(models, name) for name in models.__all__}
    assert exported_models == {mapper.class_ for mapper in Base.registry.mappers}
    assert set(Base.metadata.tables) == set(PUBLIC_MODELS.values())
    for name, table_name in PUBLIC_MODELS.items():
        model = getattr(models, name)
        assert issubclass(model, Base)
        assert model.__table__ is Base.metadata.tables[table_name]


def test_database_initialization_resolves_cross_domain_foreign_keys(database):
    # Repeated startup must also be safe after all domain modules have registered.
    database.initialize()
    schema = inspect(database.engine)
    assert set(schema.get_table_names()) == set(PUBLIC_MODELS.values())
    foreign_keys = {
        (table_name, key["constrained_columns"][0], key["referred_table"], key["referred_columns"][0])
        for table_name in schema.get_table_names()
        for key in schema.get_foreign_keys(table_name)
    }
    assert foreign_keys == {
        ("exports", "search_id", "searches", "id"),
        ("gallery_index_artifacts", "version_id", "gallery_versions", "id"),
        ("gallery_versions", "gallery_id", "galleries", "id"),
        ("import_items", "session_id", "import_sessions", "id"),
        ("import_sessions", "gallery_id", "galleries", "id"),
        ("import_sessions", "job_id", "jobs", "id"),
        ("import_sessions", "version_id", "gallery_versions", "id"),
        ("observation_digests", "observation_id", "observations", "id"),
        ("observations", "version_id", "gallery_versions", "id"),
        ("review_annotations", "review_id", "reviews", "id"),
        ("reviews", "search_id", "searches", "id"),
        ("searches", "gallery_id", "galleries", "id"),
        ("searches", "gallery_version_id", "gallery_versions", "id"),
    }
    cascading_tables = {
        table_name
        for table_name in schema.get_table_names()
        for key in schema.get_foreign_keys(table_name)
        if key["options"].get("ondelete") == "CASCADE"
    }
    assert cascading_tables == {"gallery_index_artifacts", "observation_digests"}


@pytest.mark.parametrize(
    ("table_name", "columns"),
    [
        ("gallery_versions", ("gallery_id", "number")),
        ("searches", ("workspace", "idempotency_key")),
        ("import_sessions", ("workspace", "idempotency_key")),
        ("import_items", ("session_id", "image_id")),
    ],
)
def test_domain_identity_constraints_are_created(database, table_name, columns):
    constraints = inspect(database.engine).get_unique_constraints(table_name)
    assert [tuple(constraint["column_names"]) for constraint in constraints] == [columns]
