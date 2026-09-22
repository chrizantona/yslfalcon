# Источники шрифтов

Шрифты обслуживаются локально, без внешних CDN. Новые файлы загружены 2026-09-23 из официального [Google Fonts, commit `e44c4b011a820c2cbe2fd2cfa8052037d7edb571`](https://github.com/google/fonts/tree/e44c4b011a820c2cbe2fd2cfa8052037d7edb571). Лицензия всех семейств — SIL Open Font License 1.1; тексты лицензий хранятся рядом с TTF.

| Семейство     | Локальный файл            | Начертания               | Источники                                                                                                                                                                                                                                                  |
| ------------- | ------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manrope       | `Manrope-Variable.ttf`    | variable `wght` 200–800  | [TTF](https://raw.githubusercontent.com/google/fonts/e44c4b011a820c2cbe2fd2cfa8052037d7edb571/ofl/manrope/Manrope%5Bwght%5D.ttf), [OFL](https://raw.githubusercontent.com/google/fonts/e44c4b011a820c2cbe2fd2cfa8052037d7edb571/ofl/manrope/OFL.txt)       |
| Unbounded     | `Unbounded-Variable.ttf`  | variable `wght` 200–900  | [TTF](https://raw.githubusercontent.com/google/fonts/e44c4b011a820c2cbe2fd2cfa8052037d7edb571/ofl/unbounded/Unbounded%5Bwght%5D.ttf), [OFL](https://raw.githubusercontent.com/google/fonts/e44c4b011a820c2cbe2fd2cfa8052037d7edb571/ofl/unbounded/OFL.txt) |
| IBM Plex Mono | `IBMPlexMono-Regular.ttf` | статическое Regular, 400 | [TTF](https://raw.githubusercontent.com/google/fonts/e44c4b011a820c2cbe2fd2cfa8052037d7edb571/ofl/ibmplexmono/IBMPlexMono-Regular.ttf), [локальная OFL](IBMPlexMono-OFL.txt)                                                                               |

Существующие IBM Plex Mono Regular и его OFL сохранены без изменений. TTF побайтно совпадает с указанным источником Google Fonts; локальная лицензия сохранена из прежней поставки.

Проверено fontTools 4.60.2: реальные таблицы `fvar`/`OS/2`, Unicode `cmap` и все 66 букв русского алфавита, включая Ё/ё. Количество символов блока U+0400–U+04FF: Manrope — 104; Unbounded — 98; IBM Plex Mono — 168.

| Файл                      | Размер, байт | SHA-256                                                            |
| ------------------------- | -----------: | ------------------------------------------------------------------ |
| `Manrope-Variable.ttf`    |       164700 | `3ae11c49db0455a3cc33e37d380f20fdb8c7f8b41dc07625c177e3d87a9d6ae6` |
| `Manrope-OFL.txt`         |         4387 | `58172e0c0fac2cda8a37b348164bb55e44b0e69051e557e92b1d3f6910141f7b` |
| `Unbounded-Variable.ttf`  |       778272 | `323b511be380c8d474ef030686b71aedde501f8d9cd46da558b7c40454372c3f` |
| `Unbounded-OFL.txt`       |         4392 | `31e5d4e83955e7103c34570dd49b0570ef490800bd65b42923c0dd02445263b3` |
| `IBMPlexMono-Regular.ttf` |       135580 | `6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46` |
| `IBMPlexMono-OFL.txt`     |         4362 | `d741e57d5f865e294df801f96b7b5161a88b211df65887e4358d271c9fc5fb4f` |
