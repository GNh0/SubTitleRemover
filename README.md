# SubTitleRemover

Windows 명령줄용 독립 자막 제거기입니다. 입력 파일을 수정하지 않고 새로운 MP4를 만듭니다.

```powershell
SubTitleRemover.exe "C:\video\input.mp4" "C:\video\clean.mp4"
```

소스에서 실행할 때는 Python 3.12 가상환경에 `requirements.txt`를 설치한 뒤 다음처럼 실행합니다.

```powershell
python remove_subtitles.py "C:\video\input.mp4" "C:\video\clean.mp4"
```

아래 구성요소가 실행 파일 또는 스크립트와 같은 폴더의 `third_party`에 필요합니다.

- `third_party/ffmpeg9/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe`
- `third_party/ffmpeg9/ffmpeg-9.0.1-essentials_build/bin/ffprobe.exe`
- `third_party/models/inpainting_lama_2025jan.onnx`

OCR은 화면 아래쪽 중앙의 넓은 글자를 찾습니다. 원본 크기에서 놓친 작은 글자는 화면을 3배 확대한 뒤 다시 찾습니다. 검출된 글자의 밝은 획과 외곽선을 픽셀 단위로 가리며, 밝은 배경의 흰 글자는 주변과의 밝기 차이로도 찾습니다. 같은 장면에 자막 없는 프레임이 있으면 그 프레임을 참조합니다. 참조할 수 없으면 LaMa 모델로 복원합니다. 복원 뒤 글자 자국이 많이 남으면 마스크를 주변 12픽셀(360p 기준)까지 확장해 한 번 더 보정하고, 장면과 주변 픽셀이 맞는 이웃 프레임에서는 보정된 결과를 재사용합니다. 원본에 오디오가 있으면 출력에도 보존합니다.

`--start 60 --end 120`으로 처리 구간을 지정하고 `--band-start 0.5`로 자막 탐색 영역을 조절할 수 있습니다. `--chunk-seconds 10`은 긴 구간을 독립적인 조각으로 처리하고 다시 합칩니다. 조각 경계의 영상 프레임과 오디오를 검사하지만, 분할 자체가 자막 제거 품질을 높이지는 않습니다. 장면 간 참조를 최대한 활용하려면 기본값인 단일 구간 처리를 사용하세요.

CUDA 그래픽카드와 별도 Python 환경에 `torch`, `Pillow`, `simple-lama-inpainting`이 설치되어 있다면 Big-LaMa를 복원 엔진으로 선택할 수 있습니다. 이 경우 OCR과 영상 입출력은 기본 프로그램이 담당하고, `big_lama_worker.py`가 지정된 Python에서 모델을 한 번만 불러 여러 프레임을 처리합니다. 모델 파일과 GPU Python 환경은 릴리스 압축 파일에 포함되지 않습니다.

```powershell
SubTitleRemover.exe "C:\video\input.mp4" "C:\video\clean.mp4" `
  --big-lama-python "C:\gpu-python\python.exe" `
  --big-lama-model "C:\models\big-lama.pt"
```

프로그램은 같은 장면의 깨끗한 원본 프레임을 우선 참조합니다. 그런 프레임이 없으면 선택한 LaMa 엔진으로 복원하고, 글자 잔여량이 많을 때만 마스크를 확장해 다시 처리합니다. 주변 픽셀이 일치하는 이웃 프레임에서는 깨끗하게 복원된 프레임을 재사용합니다. 이 선택은 프레임별로 이루어지며 장면 경계를 넘어 복사하지 않습니다.

자막 구간마다 CPU/GPU 복원 엔진을 선택하려면 JSON 전략 파일을 지정합니다. 시간은 원본 영상의 초 단위이며, 각 구간은 겹치지 않아야 합니다. 가급적 자막 문장이나 장면이 바뀌는 지점에서 엔진을 전환하세요. 같은 장면에서 주변 화소가 충분히 일치하면, 엔진이 바뀌어도 앞 프레임의 깨끗한 복원 부분을 이어서 사용합니다.

```json
{
  "default": "gpu",
  "segments": [
    {"start": 10.0, "end": 14.0, "method": "cpu"}
  ]
}
```

```powershell
SubTitleRemover.exe "C:\video\input.mp4" "C:\video\clean.mp4" `
  --strategy-map "C:\video\strategies.json" `
  --big-lama-python "C:\gpu-python\python.exe" `
  --big-lama-model "C:\models\big-lama.pt"
```

완료 후 출력 전체를 디코딩하고 프레임 수, 길이, 오디오를 검사합니다. 출력 옆에 `.restoration.json` 보고서를 씁니다. 새 출력 경로만 받으며 기존 파일을 덮어쓰지 않습니다. 성공한 작업의 임시 파일은 삭제하고 실패한 작업은 진단용으로 남깁니다.

화면 안의 간판이나 제목도 자막과 위치와 모양이 같으면 잘못 제거될 수 있습니다. 자막이 검출되지 않거나 복원한 배경이 흐려지는 장면도 있으므로 결과를 검수하세요. 작은 글자와 왼쪽·오른쪽 장면 표시는 감지 대상에서 제외합니다. 움직임이 크거나 자막 뒤의 원래 영상 정보가 전혀 없는 장면은 완전한 복원을 보장할 수 없습니다.

모델 출처와 구성요소 라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md)에 정리했습니다.
