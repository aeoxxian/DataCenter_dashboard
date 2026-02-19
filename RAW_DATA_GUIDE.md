# 원본 데이터(Raw Data) 제공 및 활용 방안

Streamlit Cloud는 용량 제한(약 1GB~3GB 램, 파일당 100MB)이 있어 원본 데이터(19GB)를 직접 올리는 것은 불가능합니다.
하지만 사용자가 원본 데이터를 필요로 할 때, 이를 해결할 수 있는 3가지 방안을 제안합니다.

---

## 1. 다운로드 링크 제공 (가장 추천)

가장 현실적이고 쉬운 방법입니다. 원본 데이터는 외부 스토리지(Google Drive, Dropbox, AWS S3 등)에 올려두고, Streamlit 앱에서는 **다운로드 링크**만 제공하는 것입니다.

### 구현 방법
1. 원본 데이터를 압축하여 클라우드 스토리지에 업로드하고 '공유 링크'를 생성합니다.
2. `streamlit_app.py`에 링크 버튼을 추가합니다.

```python
st.markdown("### 💾 원본 데이터 다운로드")
st.markdown("이 실험의 전체 Raw Data가 필요하신가요?")
st.link_button("Download Raw CSV (150MB)", "https://your-storage-link.com/file.csv")
```

이렇게 하면 앱은 가볍게 유지하면서, 필요한 사람만 원본을 받아볼 수 있습니다.

---

## 2. 외부 스토리지 실시간 연동 (S3 / Google Cloud)

사용자가 원본 데이터를 굳이 파일로 받지 않고 화면에서 보고 싶어 한다면, 데이터가 필요할 때만 실시간으로 클라우드에서 읽어오는 방법입니다.

### 구현 방법
1. **AWS S3**에 데이터를 업로드합니다.
2. Streamlit Secrets에 AWS 키를 등록합니다.
3. `boto3` 라이브러리로 특정 파일만 읽어옵니다.

```python
import boto3
import pandas as pd
from io import StringIO

s3 = boto3.client('s3', ...)
obj = s3.get_object(Bucket='my-bucket', Key='experiments/exp1/gpu_samples.csv')
df = pd.read_csv(obj['Body']) # 메모리 주의!
```

**단점**: 
- 구현이 복잡합니다.
- 대용량 파일을 읽을 때 로딩 시간이 길어지고 Streamlit 메모리 한도를 초과할 수 있습니다.

---

## 3. 다단계 샘플링 (Medium & Heavy)

현재 적용된 `build_lite_dataset.py`는 2000개 포인트로 줄였지만, **상세 분석용(Zoom-in)**으로 10000개 포인트 정도의 '중간 크기' 데이터를 하나 더 만드는 방법입니다.

### 구현 방법
- `lite` (2000 points): 기본 로딩용 (빠름)
- `medium` (10000 points): 자세히 보기용 (버튼 클릭 시 로딩)

이렇게 하면 GitHub 용량 제한(100MB) 내에서 훨씬 더 정밀한 데이터를 제공할 수 있습니다.

---

## ✅ 결론
- **일반적인 사용자**를 위해서는 현재의 **Lite 데이터**만으로도 충분합니다.
- **연구자/개발자**를 위해 원본이 필요하다면 **1번(다운로드 링크)** 방식이 가장 효율적입니다.
