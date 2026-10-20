"""Stable transport-neutral application errors."""


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def image_validation_message(error: ValueError) -> str:
    messages = {
        "Image must contain 1–20 MiB of JPEG or PNG data": "Нужен непустой JPEG или PNG размером не более 20 МиБ",
        "Only single-frame JPEG and PNG are supported": "Поддерживаются только однокадровые JPEG и PNG",
        "Image exceeds 24 megapixels": "Разрешение изображения превышает 24 мегапикселя",
        "Normalize EXIF orientation and bounding-box coordinates before upload": "Перед загрузкой примените поворот EXIF к изображению и пересчитайте координаты рамки",
        "BBox must be [x, y, width, height]": "Рамка должна иметь вид [x, y, ширина, высота]",
        "BBox coordinates must be finite integers": "Координаты рамки должны быть конечными целыми числами",
        "BBox width and height must be positive": "Ширина и высота рамки должны быть положительными",
        "BBox does not intersect the image": "Рамка находится за пределами изображения",
    }
    return messages.get(
        str(error), "Изображение повреждено, имеет неподдерживаемый формат или превышает ограничения"
    )
