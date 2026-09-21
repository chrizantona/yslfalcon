import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { bboxFromPoints, formatBytes, validBBox } from '../domain';
import type { BBox } from '../types';
import { Button, Icon, Message } from './ui';

export interface ImageInput {
  file: File;
  url: string;
  width: number;
  height: number;
  bbox: BBox;
}
export function CropEditor({
  value,
  onChange,
  maxBytes,
}: {
  value: ImageInput | null;
  onChange: (value: ImageInput | null) => void;
  maxBytes?: number;
}) {
  const input = useRef<HTMLInputElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const dragStart = useRef<[number, number] | null>(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const objectUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [],
  );
  const chooseFile = (file?: File) => {
    if (!file) return;
    if (!/\.(png|jpe?g)$/i.test(file.name)) {
      setError('Выберите изображение JPEG или PNG.');
      return;
    }
    if (maxBytes && file.size > maxBytes) {
      setError(`Размер файла превышает ${formatBytes(maxBytes)}.`);
      return;
    }
    setError('');
    const url = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = url;
      onChange({
        file,
        url,
        width: probe.naturalWidth,
        height: probe.naturalHeight,
        bbox: [0, 0, probe.naturalWidth, probe.naturalHeight],
      });
    };
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      setError('Браузер не может прочитать это изображение. Проверьте файл.');
    };
    probe.src = url;
  };
  const point = (event: PointerEvent): [number, number] => {
    const rect = image.current!.getBoundingClientRect();
    return [
      Math.max(
        0,
        Math.min(value!.width, ((event.clientX - rect.left) / rect.width) * value!.width),
      ),
      Math.max(
        0,
        Math.min(value!.height, ((event.clientY - rect.top) / rect.height) * value!.height),
      ),
    ];
  };
  const move = (event: PointerEvent) => {
    if (!value || !dragStart.current) return;
    onChange({
      ...value,
      bbox: bboxFromPoints(dragStart.current, point(event), value.width, value.height),
    });
  };
  return (
    <div className="crop-editor">
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png"
        className="visually-hidden"
        aria-label="Загрузить изображение запроса"
        onChange={(event) => {
          chooseFile(event.target.files?.[0]);
          event.target.value = '';
        }}
      />
      {value ? (
        <>
          <div className="source-image-area">
            <div
              className="crop-stage"
              onPointerDown={(event) => {
                if (event.button !== 0) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                dragStart.current = point(event);
              }}
              onPointerMove={move}
              onPointerUp={(event) => {
                move(event);
                dragStart.current = null;
              }}
              onPointerCancel={() => {
                dragStart.current = null;
              }}
            >
              <img
                ref={image}
                src={value.url}
                alt="Исходное изображение. Выделите автомобиль указателем или задайте координаты ниже."
                draggable={false}
              />
              <div
                className="bbox-overlay"
                style={{
                  left: `${(value.bbox[0] / value.width) * 100}%`,
                  top: `${(value.bbox[1] / value.height) * 100}%`,
                  width: `${(value.bbox[2] / value.width) * 100}%`,
                  height: `${(value.bbox[3] / value.height) * 100}%`,
                }}
              >
                <span>ОБЛАСТЬ ПОИСКА</span>
                <i />
                <i />
                <i />
                <i />
              </div>
            </div>
          </div>
          <div className="file-caption">
            <span title={value.file.name}>{value.file.name}</span>
            <span>
              {value.width} × {value.height}
            </span>
          </div>
          <div className="crop-tools">
            <span>Выделите автомобиль на снимке</span>
            <Button
              tone="ghost"
              onClick={() => onChange({ ...value, bbox: [0, 0, value.width, value.height] })}
            >
              Весь кадр
            </Button>
          </div>
          <div className="bbox-fields">
            {['X', 'Y', 'Ширина', 'Высота'].map((label, index) => (
              <label key={label}>
                <span>{label}</span>
                <input
                  type="number"
                  min={index < 2 ? 0 : 1}
                  step="1"
                  value={value.bbox[index]}
                  onChange={(event) => {
                    const bbox = [...value.bbox] as BBox;
                    bbox[index] = Number(event.target.value);
                    onChange({ ...value, bbox });
                  }}
                />
              </label>
            ))}
          </div>
          {!validBBox(value.bbox, value.width, value.height) && (
            <Message tone="error">
              Рамка должна целиком находиться внутри изображения и иметь положительные целые
              размеры.
            </Message>
          )}
          <div className="crop-preview-row">
            <svg
              className="crop-preview"
              viewBox={value.bbox.join(' ')}
              role="img"
              aria-label="Вырезка, которая будет отправлена на поиск"
            >
              <image href={value.url} width={value.width} height={value.height} />
            </svg>
            <div>
              <strong>В поиск попадёт эта область</strong>
              <p>Координаты в пикселях исходного кадра</p>
            </div>
          </div>
          <Button icon="upload" onClick={() => input.current?.click()} className="full-width">
            Заменить изображение
          </Button>
        </>
      ) : (
        <button
          className={`upload-zone ${dragging ? 'is-dragging' : ''}`}
          onClick={() => input.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            chooseFile(event.dataTransfer.files[0]);
          }}
        >
          <div className="upload-illustration" aria-hidden="true">
            <div className="illustration-frame back" />
            <div className="illustration-frame">
              <Icon name="image" size={38} />
              <span />
              <i />
              <i />
              <i />
              <i />
            </div>
            <div className="illustration-add">
              <Icon name="plus" size={18} />
            </div>
          </div>
          <strong>Добавьте изображение</strong>
          <span>
            Перетащите файл сюда
            <br />
            или выберите на устройстве
          </span>
          <small>JPEG, PNG{maxBytes ? ` · до ${formatBytes(maxBytes)}` : ''}</small>
        </button>
      )}
      {error && <Message tone="error">{error}</Message>}
    </div>
  );
}
