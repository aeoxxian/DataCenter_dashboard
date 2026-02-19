# 대용량 원본 데이터(19GB) 무료 호스팅 및 연결 가이드

GitHub 용량 제한을 우회하여 19GB 원본 데이터를 사용자에게 제공하고 싶다면, **Hugging Face Datasets**를 사용하는 것이 가장 좋습니다. (무제한 용량, 무료, 고속 다운로드 지원)

## 1 단계: Hugging Face에 데이터 올리기

1. **가입 및 저장소 생성**
   - [Hugging Face](https://huggingface.co/) 가입
   - 우측 상단 프로필 -> **New Dataset** 클릭
   - 저장소 이름 입력 (예: `my-power-data`) -> **Create dataset**

2. **데이터 업로드 (웹 브라우저 이용)**
   - 생성된 저장소 페이지에서 **Files and versions** 탭 클릭
   - **Add file** -> **Upload files** 클릭
   - 로컬의 `outputs` 폴더 안에 있는 실험 폴더들을 드래그 앤 드롭으로 업로드
   - **Commit changes** 클릭 (용량이 크면 시간이 좀 걸립니다)
   
   *(참고: 19GB를 한 번에 올리기 힘들다면, Git LFS를 사용해야 합니다. 웹에서는 폴더별로 나누어 올리는 것을 추천합니다.)*

## 2 단계: 다운로드 링크 URL 확인

업로드가 완료되면, 파일 하나(예: `runs/gpu_samples.csv`)를 클릭해보세요.
**"download"** 버튼에 마우스를 올리고 주소를 복사하면 다음과 같은 형식이 됩니다:

```
https://huggingface.co/datasets/{사용자명}/{저장소명}/resolve/main/outputs/{실험폴더}/{런폴더}/gpu_samples.csv
```

여기서 앞부분 `https://huggingface.co/.../resolve/main/` 이 **Base URL**이 됩니다.

## 3 단계: Streamlit 앱에 연결

`streamlit_app.py` 파일(약 370번째 줄)을 열어 주석 처리된 부분을 수정하세요.

```python
# 수정 전
# raw_url = f"https://huggingface.co/datasets/..."
# st.link_button("Download Full Raw Data (External)", raw_url)

# 수정 후 (예시)
relative_path = r["path"].relative_to(OUTPUTS_ROOT).as_posix() 
base_url = "https://huggingface.co/datasets/hyunwoo/my-power-data/resolve/main"
raw_url = f"{base_url}/{relative_path}/gpu_samples.csv"

st.link_button("☁️ Download Full Raw Data (19GB)", raw_url)
```

이렇게 하면 사용자가 버튼을 클릭했을 때 Hugging Face에서 원본 데이터를 바로 다운로드받을 수 있습니다!
