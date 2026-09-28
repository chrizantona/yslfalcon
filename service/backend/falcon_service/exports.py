"""Portable search reports from immutable results, with independent reviews."""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import BinaryIO
from xml.sax.saxutils import escape

from sqlalchemy import select

from .catalog import scoped
from .db import Database
from .errors import ServiceError
from .models import Observation, Search
from .search import search_view
from .storage import ArtifactStore

MAX_CASE_SOURCE_BYTES = 256 * 1024 * 1024
MAX_CASE_ARCHIVE_BYTES = 256 * 1024 * 1024
CASE_SPOOL_BYTES = 8 * 1024 * 1024
PREVIEW_REASON = "preview_descriptor_is_not_vehicle_reidentification"


def _pdf_notice(result: dict) -> str:
    if result.get("calibration_reason") == PREVIEW_REASON:
        return (
            "Модель не подключена. Этот отчёт демонстрирует работу сервиса и не подтверждает "
            "идентичность автомобилей."
        )
    return (
        f"Ранжирование сформировано моделью {result['model_id']}. Оценки сходства не являются "
        "вероятностями; подтверждение совпадения допустимо только при применимой версионированной "
        "калибровке."
    )


def _pdf_decision_label(result: dict) -> str:
    if result.get("calibration_reason") == PREVIEW_REASON:
        return "Не определено: работает техническая заглушка"
    return {
        "candidates": "Кандидат найден",
        "no_confident_match": "Уверенное совпадение не найдено",
        "calibration_unavailable": "Ранжирование без решения: калибровка недоступна",
    }.get(result.get("decision"), str(result.get("decision", "Не определено")))


def make_passport(db: Database, store: ArtifactStore, workspace: str, search_id: str) -> dict:
    with db.transaction() as session:
        search = scoped(session, Search, search_id, workspace)
        result = search_view(session, search)
        observations = list(
            session.scalars(
                select(Observation).where(
                    Observation.version_id == search.gallery_version_id,
                    Observation.workspace == workspace,
                )
            )
        )
    by_id = {item.id: item for item in observations}
    originals = [
        {"kind": "query", "sha256": hashlib.sha256(store.read(search.query["image_key"])).hexdigest()}
    ]
    for candidate in result["ranked_candidates"]:
        observation = by_id[candidate["observation_id"]]
        originals.append(
            {
                "kind": "candidate",
                "observation_id": observation.id,
                "sha256": hashlib.sha256(store.read(observation.image_key)).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "purpose": "visual_search_report",
        "search": result,
        "integrity": {"algorithm": "sha256", "original_images": originals},
        "notice": "PREVIEW: no trained vehicle ReID model is connected. Results demonstrate the service workflow only. Operator reviews are separate from algorithmic results; a similarity score is not a probability.",
    }


def render_export(passport: dict, format: str, db: Database, store: ArtifactStore, workspace: str) -> bytes:
    if format == "json":
        return json.dumps(passport, ensure_ascii=False, indent=2, allow_nan=False).encode()
    if format == "csv":
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            [
                "search_id",
                "gallery_version_id",
                "model_id",
                "decision",
                "rank",
                "observation_id",
                "cosine",
                "score",
                "accepted",
            ]
        )
        result = passport["search"]
        for candidate in result["ranked_candidates"]:
            writer.writerow(
                [
                    result["id"],
                    result["gallery_version_id"],
                    result["model_id"],
                    result["decision"],
                    candidate["rank"],
                    candidate["observation_id"],
                    candidate["cosine"],
                    candidate["score"],
                    candidate["accepted"],
                ]
            )
        return output.getvalue().encode("utf-8-sig")
    if format == "pdf":
        return render_pdf(passport, db, store, workspace)
    raise ServiceError("invalid_export_format", "Неизвестный формат экспорта", 400)


def _image_extension(data: bytes) -> str:
    return "png" if data.startswith(b"\x89PNG") else "jpg"


def _offline_viewer(
    passport: dict,
    candidate_paths: dict[str, dict[str, str]],
    query_paths: dict[str, str],
) -> str:
    result = passport["search"]
    annotation_labels = {"match": "совпадает", "conflict": "противоречит", "unclear": "неясно"}
    annotation_classes = {"match": "mark-match", "conflict": "mark-conflict", "unclear": "mark-unclear"}
    annotations = [
        {
            **item,
            "review_number": number,
            "review_observation_id": review.get("observation_id"),
        }
        for number, review in enumerate(result["reviews"], 1)
        for item in review.get("annotations", [])
    ]

    def markers(image_ref: str, surface: str) -> str:
        return "".join(
            "<span class='marker "
            + annotation_classes[item["kind"]]
            + f"' style='left:{item['x'] * 100:.4f}%;top:{item['y'] * 100:.4f}%' "
            + "title='Заключение #"
            + str(item["review_number"])
            + " по кандидату "
            + escape(item["review_observation_id"] or "всему поиску")
            + ": "
            + escape(annotation_labels[item["kind"]])
            + "'>R"
            + str(item["review_number"])
            + "</span>"
            for item in annotations
            if item["image_ref"] == image_ref and item["surface"] == surface
        )

    def photo(path: str, image_ref: str, surface: str, alt: str) -> str:
        return (
            f"<div class='photo'><img src='{path}' alt='{escape(alt)}'>"
            + markers(image_ref, surface)
            + "</div>"
        )

    candidate_cards = []
    for candidate in result["ranked_candidates"]:
        observation_id = candidate["observation_id"]
        label = candidate.get("external_id") or observation_id
        candidate_cards.append(
            "<article><header><b>Кандидат №"
            + str(candidate["rank"])
            + "</b><code>"
            + escape(label)
            + "</code></header>"
            + photo(
                candidate_paths[observation_id]["crop"],
                observation_id,
                "crop",
                f"Кандидат №{candidate['rank']}",
            )
            + "<details><summary>Исходный кадр и пометки</summary>"
            + photo(
                candidate_paths[observation_id]["original"],
                observation_id,
                "original",
                f"Исходный кадр кандидата №{candidate['rank']}",
            )
            + "</details><p>Косинус "
            + f"{candidate['cosine']:.5f} · score {candidate['score']:.5f}"
            + "</p></article>"
        )
    verdict_labels = {
        "match": "совпадение",
        "different": "другое ТС",
        "uncertain": "недостаточно данных",
    }
    reviews = []
    for number, review in enumerate(result["reviews"], 1):
        pair = review.get("observation_id") or "весь поиск"
        reviews.append(
            f"<li><b>Заключение #{number}: {escape(verdict_labels[review['verdict']])}</b> · "
            + escape(review["created_at"])
            + "<br>Пара: запрос ↔ "
            + escape(pair)
            + "<br>"
            + escape(review.get("note") or "Без комментария")
            + f"<br><small>Координатных пометок: {len(review.get('annotations', []))}</small></li>"
        )
    review_html = "".join(reviews) or "<li>Заключения оператора не добавлены.</li>"
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Фалькон · дело {escape(result["id"])}</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0d0d0e;color:#f5f5f5;
font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:32px}}h1{{font-size:25px}}
.notice{{padding:14px;border:1px solid #555;background:#171719}}.board{{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}}article{{border:1px solid #3a3a3d;
background:#151517}}header{{display:flex;justify-content:space-between;gap:12px;padding:12px}}code,small{{color:#aaa}}
.photo{{position:relative;background:#080809;overflow:hidden}}.photo img{{display:block;width:100%;height:auto}}
article p{{padding:0 12px;color:#aaa}}.marker{{position:absolute;
min-width:24px;height:24px;padding:0 3px;border:2px solid white;border-radius:12px;transform:translate(-50%,-50%);
display:grid;place-items:center;color:#111;font:bold 9px/1 monospace;
box-shadow:0 0 0 4px #0008}}.mark-match{{background:#7bd88f}}.mark-conflict{{background:#ff7373}}
.mark-unclear{{background:#f4cf65}}dl{{display:grid;grid-template-columns:180px 1fr}}dt{{color:#888}}
details{{border-top:1px solid #333}}summary{{cursor:pointer;padding:10px 12px;color:#bbb}}
li{{margin:12px 0}}@media(max-width:850px){{.board{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style></head><body><main><p>ФАЛЬКОН / OFFLINE CASE</p><h1>Доказуемое дело визуального поиска</h1>
<p class="notice">{escape(passport["notice"])}</p><dl><dt>Поиск</dt><dd>{escape(result["id"])}</dd>
<dt>Версия галереи</dt><dd>{escape(result["gallery_version_id"])}</dd><dt>Модель</dt>
<dd>{escape(result["model_id"])}</dd><dt>Создано</dt><dd>{escape(result["created_at"])}</dd></dl>
<section class="board"><article><header><b>Запрос</b><code>BBox {escape(str(result["query"]["bbox"]))}</code></header>
{photo(query_paths["crop"], "query", "crop", "Запрос")}
<details><summary>Исходный кадр и пометки</summary>{photo(query_paths["original"], "query", "original", "Исходный кадр запроса")}</details>
<p>Проверенная область исходного кадра</p></article>{"".join(candidate_cards)}</section>
<h2>Заключения оператора</h2><ul>{review_html}</ul><p><a href="report.pdf">Открыть PDF</a> ·
<a href="report.json">Открыть JSON</a> · контроль целостности: <code>python3 verify.py</code></p>
</main></body></html>"""


def render_case(
    passport: dict,
    db: Database,
    store: ArtifactStore,
    workspace: str,
    output: BinaryIO,
) -> int:
    """Stream a bounded, self-contained case into a seekable output."""
    from .inference import crop_image

    result = passport["search"]
    with db.transaction() as session:
        search = scoped(session, Search, result["id"], workspace)
        observations = {
            item.id: item
            for item in session.scalars(
                select(Observation).where(
                    Observation.version_id == search.gallery_version_id,
                    Observation.workspace == workspace,
                )
            )
        }
    manifest_files: list[dict] = []
    source_bytes = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def add(path: str, data: bytes) -> None:
            nonlocal source_bytes
            source_bytes += len(data)
            if source_bytes > MAX_CASE_SOURCE_BYTES:
                raise ServiceError("case_too_large", "Размер offline-дела превышает 256 МиБ", 413)
            archive.writestr(path, data)
            manifest_files.append(
                {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )

        query = store.read(search.query["image_key"])
        query_paths = {
            "original": f"images/query/original.{_image_extension(query)}",
            "crop": "images/query/crop.png",
        }
        add(query_paths["original"], query)
        add(query_paths["crop"], crop_image(query, search.query["bbox"]))
        candidate_paths = {}
        for candidate in result["ranked_candidates"]:
            observation = observations[candidate["observation_id"]]
            image = store.read(observation.image_key)
            folder = f"images/candidates/{candidate['rank']:02d}-{observation.id}"
            paths = {
                "original": f"{folder}/original.{_image_extension(image)}",
                "crop": f"{folder}/crop.png",
            }
            add(paths["original"], image)
            add(paths["crop"], crop_image(image, observation.bbox))
            candidate_paths[observation.id] = paths
        add(
            "report.json",
            json.dumps(passport, ensure_ascii=False, indent=2, allow_nan=False).encode(),
        )
        add("report.pdf", render_pdf(passport, db, store, workspace))
        add(
            "viewer.html",
            _offline_viewer(passport, candidate_paths, query_paths).encode(),
        )
        add(
            "README.txt",
            (
                "Фалькон · автономное дело\n\n"
                "Откройте viewer.html в браузере. Для проверки целостности выполните "
                "python3 verify.py в этой папке. Результаты preview не подтверждают идентичность ТС.\n"
            ).encode(),
        )
        add(
            "verify.py",
            """import hashlib
import json
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
errors = []
for item in manifest["files"]:
    path = (root / item["path"]).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or path.stat().st_size != item["size"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
    ):
        errors.append(item["path"])
print("OK: all files verified" if not errors else "FAILED: " + ", ".join(errors))
sys.exit(bool(errors))
""".encode(),
        )
        manifest = {
            "schema_version": 1,
            "algorithm": "sha256",
            "search_id": result["id"],
            "files": sorted(manifest_files, key=lambda item: item["path"]),
        }
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode(),
        )
    return source_bytes


def render_pdf(passport: dict, db: Database, store: ArtifactStore, workspace: str) -> bytes:
    from PIL import Image as PILImage
    from PIL import ImageDraw
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as ReportImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from .inference import crop_image

    buffer = io.BytesIO()
    font_name = "Helvetica"
    for font in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(font).is_file():
            font_name = "FalconReport"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, font))
            break
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = font_name
    normal = styles["BodyText"]
    normal.fontSize = 9
    normal.leading = 13
    normal.textColor = colors.HexColor("#343d4b")
    styles["Title"].fontSize = 25
    styles["Title"].leading = 31
    styles["Title"].alignment = 0
    styles["Title"].textColor = colors.HexColor("#202b3c")
    styles["Heading2"].fontSize = 13
    styles["Heading2"].leading = 18
    styles["Heading2"].textColor = colors.HexColor("#202b3c")
    result = passport["search"]
    annotations = [
        {**item, "review_number": number}
        for number, review in enumerate(result["reviews"], 1)
        for item in review.get("annotations", [])
    ]

    def paragraph(value):
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), normal)

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Фалькон - отчёт по визуальному поиску",
        author="Фалькон",
    )
    flow = [
        paragraph("ФАЛЬКОН / ВИЗУАЛЬНЫЙ ПОИСК"),
        Spacer(1, 8),
        Paragraph("Отчёт по поиску", styles["Title"]),
        paragraph(_pdf_notice(result)),
        Spacer(1, 12),
    ]
    for label, value in [
        ("Идентификатор поиска", result["id"]),
        ("Время (UTC)", result["created_at"]),
        ("Версия галереи", result["gallery_version_id"]),
        ("Решение алгоритма", _pdf_decision_label(result)),
        ("Принятых кандидатов", len(result["accepted_candidates"])),
    ]:
        flow.append(paragraph(f"{label}: {value}"))
    flow.append(Spacer(1, 12))
    with db.transaction() as session:
        search = scoped(session, Search, result["id"], workspace)
        observations = {
            item.id: item
            for item in session.scalars(
                select(Observation).where(
                    Observation.version_id == search.gallery_version_id, Observation.workspace == workspace
                )
            )
        }

    def annotated_surface(data: bytes, bbox: list[int], image_ref: str, surface: str) -> bytes:
        source = data if surface == "original" else crop_image(data, bbox)
        with PILImage.open(io.BytesIO(source)) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        line = max(2, min(image.size) // 160)
        if surface == "original":
            x, y, width, height = bbox
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#f4cf65", width=line)
        marker_colors = {"match": "#7bd88f", "conflict": "#ff7373", "unclear": "#f4cf65"}
        radius = max(5, min(image.size) // 45)
        for item in annotations:
            if item["image_ref"] != image_ref or item["surface"] != surface:
                continue
            center_x = min(image.width - 1, max(0, round(item["x"] * (image.width - 1))))
            center_y = min(image.height - 1, max(0, round(item["y"] * (image.height - 1))))
            bounds = (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            )
            draw.ellipse(bounds, fill=marker_colors[item["kind"]], outline="white", width=line)
            draw.text(
                (center_x, center_y),
                f"R{item['review_number']}",
                fill="#111111",
                anchor="mm",
            )
        rendered = io.BytesIO()
        image.save(rendered, "PNG")
        return rendered.getvalue()

    def thumbnail(data: bytes, max_width: int, max_height: int):
        image = ReportImage(io.BytesIO(data))
        scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        return image

    query_data = store.read(search.query["image_key"])
    query_images = Table(
        [
            [paragraph("Исходный кадр · BBox"), paragraph("Вырезка")],
            [
                thumbnail(
                    annotated_surface(query_data, search.query["bbox"], "query", "original"),
                    245,
                    150,
                ),
                thumbnail(
                    annotated_surface(query_data, search.query["bbox"], "query", "crop"),
                    245,
                    150,
                ),
            ],
        ],
        colWidths=[257, 258],
    )
    query_images.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    flow.extend(
        [
            Paragraph("Изображение запроса", styles["Heading2"]),
            paragraph("Рамка и координатные пометки привязаны к фактическим пикселям поверхности."),
            Spacer(1, 8),
            query_images,
            Spacer(1, 14),
        ]
    )
    flow.extend(
        [
            Paragraph("Кандидаты для сравнения", styles["Heading2"]),
            paragraph("Косинус и оценка ранжирования не являются вероятностью совпадения."),
            Spacer(1, 8),
        ]
    )
    for candidate in result["ranked_candidates"]:
        observation = observations[candidate["observation_id"]]
        candidate_data = store.read(observation.image_key)
        original = thumbnail(
            annotated_surface(candidate_data, observation.bbox, observation.id, "original"),
            155,
            110,
        )
        crop = thumbnail(
            annotated_surface(candidate_data, observation.bbox, observation.id, "crop"),
            155,
            110,
        )
        description = paragraph(
            f"Кандидат № {candidate['rank']}\nНаблюдение: {observation.id}"
            f"\nКосинус: {candidate['cosine']:.6f}\nОценка: {candidate['score']:.6f}"
            f"\nПринят алгоритмом: {'да' if candidate['accepted'] else 'нет'}"
        )
        table = Table([[original, crop, description]], colWidths=[165, 165, 185])
        table.setStyle(
            TableStyle(
                [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.lightgrey)]
            )
        )
        flow.extend([table, Spacer(1, 10)])
    flow.append(Paragraph("Заключения оператора", styles["Heading2"]))
    flow.append(paragraph("Сохранены отдельно от решения алгоритма и не изменяют выдачу."))
    if not result["reviews"]:
        flow.append(paragraph("Заключения ещё не добавлены."))
    verdicts = {"match": "Совпадает", "different": "Разные автомобили", "uncertain": "Недостаточно данных"}
    annotation_kinds = {"match": "совпадает", "conflict": "противоречит", "unclear": "неясно"}
    annotation_surfaces = {"crop": "вырезка", "original": "исходный кадр"}
    for number, review in enumerate(result["reviews"], 1):
        evidence = "\n".join(
            f"Пометка R{number}: "
            + annotation_kinds[item["kind"]]
            + f" / {annotation_surfaces[item['surface']]} / {item['image_ref']}"
            + f" / x={item['x']:.4f}, y={item['y']:.4f}"
            for item in review.get("annotations", [])
        )
        flow.append(
            paragraph(
                f"Заключение #{number} / {review['created_at']} / {verdicts[review['verdict']]}"
                f"\nНаблюдение: {review.get('observation_id') or 'весь поиск'}"
                f"\n{review['note']}" + (f"\n{evidence}" if evidence else "")
            )
        )
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("Паспорт воспроизводимости", styles["Heading2"]))
    flow.append(paragraph(f"Модель: {result['model_id']}"))
    flow.append(paragraph(f"Контрольная сумма галереи: {result['gallery_manifest_hash']}"))
    flow.append(paragraph("Контрольные суммы оригиналов (SHA-256):"))
    for item in passport["integrity"]["original_images"]:
        flow.append(paragraph(f"{item.get('observation_id', 'query')}: {item['sha256']}"))

    def footer(canvas, _document):
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(40, 22, f"Фалькон / {result['id']} / Страница {canvas.getPageNumber()}")

    document.build(flow, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
