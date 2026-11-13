# Dataset

Aligned eye-band crops for training and testing a camera smart glasses detector. The zip file is attached to the [releases](https://github.com/fattaillabs/smartglasses-detect/releases).

한국어 요약은 이 문서 끝에 있습니다.

## Contents

| Item | Count |
|---|---|
| Training crops | 20,958 (7,480 with camera smart glasses) |
| Test crops | 1,282 (389 with camera smart glasses) |
| Crop size | 512x192 pixels, JPEG |

Each crop is the band around the eyes of one face, rotated so the eyes are level and scaled so the face is 400 pixels wide. No full face, no original photo and no video frame is included.

```
data/train_list.jsonl                      one line per training crop
data/test_list.jsonl                       one line per test crop
data/crops/<source>/<label>/<video or set>/band/<id>.jpg
```

Fields in the lists:

| Field | Meaning |
|---|---|
| `id`, `dir` | The crop is at `<dir>/band/<id>.jpg` |
| `y` | 1 = wearing camera smart glasses, 0 = anything else |
| `prod` | Product name from `labels/products.json`. `none` = not smart glasses. Empty = smart glasses of an unknown model. |
| `group` | Video id or photo id. Crops of the same group share a person and a scene. |
| `source` | Where it came from: `yt_*` YouTube, `web_*` image search, `ffhq_*` FFHQ |
| `split` | `train` or `val` (training list only) |
| `kind` | Human-readable category (test list only) |

Train and test never share a video. Test videos also come from channels that do not appear in the training set.

## How the labels were made

- Ray-Ban Meta, Oakley Meta HSTN and the web photos of the first round: sample sheets reviewed by hand, with a small classifier separating frames with and without glasses.
- Newer products: frames of each video were grouped by similarity and each group was labeled by looking at one representative frame. The codes are S (the product of that video), O (other smart glasses), P (ordinary glasses), N (no glasses), X (dropped). A group that mixes scenes can carry a wrong label.
- Test labels were checked a second time in a gallery, and six were corrected (`labels/test_overrides.json`).

## Sources and licenses

The dataset as a whole is offered under CC BY-NC-SA 4.0 for non-commercial research. The parts have different origins:

| Part | Crops | Origin | Terms |
|---|---|---|---|
| `data/crops/ffhq` | 3,738 | Flickr-Faces-HQ | CC BY-NC-SA 4.0. Each photo keeps its own license. The FFHQ image number is the first part of the file name, so author and license can be looked up in the official `ffhq-dataset-v1.json`. |
| `data/crops/yt`, `data/crops/yt_test` | 17,302 | Frames of public YouTube videos | Copyright of each creator. The video id is the folder name and the first part of the file name. The full list is in `sources/youtube_videos.txt`. |
| `data/crops/web` | 1,200 | Image search results | Copyright of each owner. Source addresses are in `sources/web_urls_*.jsonl`, keyed by the first part of the file name. |

The YouTube and image search parts are redistributed without the permission of their owners, as eye-region crops for non-commercial research. If you use them, you take on the same responsibility. Do not use this dataset to identify people.

## Removal requests

If you appear in the dataset, or you own a source, and you want it removed, open an [issue](https://github.com/fattaillabs/smartglasses-detect/issues) with the file name or the source address. It will be removed from the next release.

## 한국어 요약

- 카메라 스마트안경 판별 모델의 학습·테스트용 데이터입니다. 정렬된 눈 주변 띠(512x192) 조각 22,240장과 정답표가 들어 있습니다. 얼굴 전체, 원본 사진, 영상 프레임은 없습니다.
- 학습 20,958장(카메라 스마트안경 7,480장), 테스트 1,282장(389장)입니다. 학습과 테스트는 영상이 겹치지 않고, 테스트 영상은 학습에 없는 채널에서 골랐습니다.
- 전체는 비상업 연구용 CC BY-NC-SA 4.0 입니다. FFHQ에서 나온 조각은 FFHQ의 조건을 따르고, 유튜브와 이미지 검색에서 나온 조각은 각 권리자의 저작물이며 허락 없이 눈 주변만 잘라 실었습니다. 사람을 식별하는 데 쓰지 마십시오.
- 본인이 나오거나 원본의 권리자인데 삭제를 원하면 이슈에 파일 이름이나 원본 주소를 남겨 주십시오.
