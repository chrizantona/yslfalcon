"""Portable search reports from immutable results, with independent reviews."""

import csv
import hashlib
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

from sqlalchemy import select

from .catalog import scoped
from .db import Database
from .models import Observation, Search
from .search import search_view
from .storage import ArtifactStore


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
    return render_pdf(passport, db, store, workspace)


def render_pdf(passport: dict, db: Database, store: ArtifactStore, workspace: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
        paragraph(
            "Модель не подключена. Этот отчёт демонстрирует работу сервиса и не подтверждает идентичность автомобилей."
        ),
        Spacer(1, 12),
    ]
    for label, value in [
        ("Идентификатор поиска", result["id"]),
        ("Время (UTC)", result["created_at"]),
        ("Версия галереи", result["gallery_version_id"]),
        ("Решение алгоритма", "Не определено: работает техническая заглушка"),
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

    def thumbnail(data):
        image = Image(io.BytesIO(data))
        scale = min(230 / image.imageWidth, 150 / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        return image

    flow.extend(
        [
            Paragraph("Изображение запроса", styles["Heading2"]),
            paragraph("Проверенная область изображения, использованная при поиске"),
            Spacer(1, 8),
            thumbnail(crop_image(store.read(search.query["image_key"]), search.query["bbox"])),
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
        photo = thumbnail(crop_image(store.read(observation.image_key), observation.bbox))
        description = paragraph(
            f"Кандидат № {candidate['rank']}\nНаблюдение: {observation.id}\nКосинус: {candidate['cosine']:.6f}\nОценка: {candidate['score']:.6f}\nПринят алгоритмом: {'да' if candidate['accepted'] else 'нет'}"
        )
        table = Table([[photo, description]], colWidths=[245, 270])
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
    for review in result["reviews"]:
        flow.append(
            paragraph(
                f"{review['created_at']} / {verdicts[review['verdict']]}\nНаблюдение: {review.get('observation_id') or 'весь поиск'}\n{review['note']}"
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
