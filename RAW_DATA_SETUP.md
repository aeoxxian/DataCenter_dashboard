# 대용량 원본 데이터(19GB) 자동 업로드 가이드

19GB 데이터를 일일이 드래그 앤 드롭하지 않고, 스크립트 하나로 **Hugging Face**에 올릴 수 있습니다.

## 1. 사전 준비 (토큰 발급)
1. [Hugging Face](https://huggingface.co/) 가입
2. [설정 > Access Tokens](https://huggingface.co/settings/tokens) 이동
3. **Create new token** 클릭 (Type: **Write** 선택 필수!)
4. 생성된 토큰(`hf_...`)을 복사해 두세요.

## 2. 자동 업로드 스크립트 실행
터미널에서 다음 명령어를 입력하세요:

```bash
python upload_raw_data_to_hf.py
```

스크립트가 실행되면 다음 두 가지를 물어봅니다:
1. **Token**: 방금 복사한 토큰 붙여넣기
2. **Repository ID**: 데이터를 올릴 저장소 이름 (예: `본인아이디/my-power-data`)

나머지는 스크립트가 알아서 처리합니다! (19GB 업로드에 시간이 좀 걸립니다 ☕)

## 3. Streamlit 앱 연결
업로드가 끝나면 스크립트가 **Base URL**을 출력해 줍니다.
이 주소를 복사해서 `streamlit_app.py`에 붙여넣으세요.

```python
# streamlit_app.py (약 370번째 줄)
base_url = "https://huggingface.co/datasets/본인아이디/my-power-data/resolve/main/outputs"
```
