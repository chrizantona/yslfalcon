# OSNet-AIN

`osnet_ain.py` is the unchanged upstream architecture by Kaiyang Zhou et al., from
[KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid/blob/f6ba78eceb38712266318e8c7eefde174af15606/torchreid/models/osnet_ain.py),
revision `f6ba78eceb38712266318e8c7eefde174af15606`. MIT licence: `LICENSE.osnet.txt`.

Our wrapper always constructs it with `pretrained=False`. Public weights are loaded only from
the local file pinned in DATA.md, downloaded by `scripts/fetch_external.sh osnet`, with SHA256 verification.
The upstream downloader is never invoked, and no torchreid or gdown dependency is needed.

We use the standard x1.0 feature map before pooling, the original FC and classifier are excluded.
The local wrapper adds bounded GeM, projection and separate auxiliary supervision outside the backbone.
