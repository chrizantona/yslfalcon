"""Backward-compatible public imports for the domain response contracts."""

# Re-exports intentionally retain the historical public namespace.
# ruff: noqa: F401

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .response_contracts.auth import IdentityResponse
from .response_contracts.common import (
    ERROR_RESPONSES,
    BBox,
    ErrorDetail,
    ErrorResponse,
    Items,
    Page,
    ResponseModel,
    Role,
    Timestamp,
)
from .response_contracts.galleries import (
    GalleryList,
    GalleryResponse,
    GalleryVersionDiffItem,
    GalleryVersionDiffResponse,
    GalleryVersionList,
    GalleryVersionResponse,
)
from .response_contracts.imports import (
    ImportIssue,
    ImportLimits,
    ImportResult,
    ImportSessionList,
    ImportSessionResponse,
    ImportValidation,
)
from .response_contracts.jobs import (
    JobList,
    JobResponse,
    LegacyAnalysisSizes,
    LegacyExplanationResult,
)
from .response_contracts.reports import (
    ExportResult,
    ImageDigest,
    ReportIntegrity,
    SearchReport,
)
from .response_contracts.searches import (
    Candidate,
    QueryImage,
    ReviewAnnotationResponse,
    ReviewResponse,
    SearchList,
    SearchResponse,
    SearchTimings,
)
from .response_contracts.system import (
    CalibrationList,
    CapabilitiesResponse,
    Dependencies,
    DependencyStatus,
    EngineCapabilities,
    JobCounts,
    LiveResponse,
    ModelList,
    ModelResponse,
    ModelStatus,
    Percentiles,
    ReadyResponse,
    SearchProfile,
    ServiceFeatures,
    ServiceLimits,
    SystemResponse,
    TimingSummary,
)

# Keep existing class references and pickled responses resolvable at this public path,
# including Pydantic-generated generic names such as ``Page[GalleryResponse]``.
for _model in tuple(globals().values()):
    if isinstance(_model, type) and issubclass(_model, BaseModel) and _model is not BaseModel:
        _model.__module__ = __name__
        globals()[_model.__name__] = _model
del _model
