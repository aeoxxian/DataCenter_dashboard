# 19GB 데이터 업로드 가이드 (Git LFS 방식)

스크립트 업로드가 네트워크 문제로 자꾸 실패한다면, **Git LFS(Large File Storage)**를 사용하는 것이 가장 확실합니다.
조금 번거롭지만, 끊겨도 재시도할 수 있고 훨씬 안정적입니다.

## 1. 사전 준비
1. [Hugging Face](https://huggingface.co/)에서 **New Dataset**으로 빈 저장소를 만드세요 (예: `aeoxxian/Datacenter_train`).
2. 터미널에서 Git LFS를 설치합니다:
   ```bash
   git lfs install
   ```

## 2. 저장소 복제 (Clone)
터미널에서 방금 만든 저장소를 내 컴퓨터로 가져옵니다.
*(비밀번호를 물어보면 Hugging Face Token을 입력하세요)*
```bash
# 예시 (본인 주소로 변경)
git clone https://huggingface.co/datasets/aeoxxian/Datacenter_train
cd Datacenter_train
```

## 3. 데이터 복사 및 설정
`outputs` 폴더를 여기로 복사하고, 대용량 파일 추적을 설정합니다.

```bash
# 1. 데이터 복사
cp -r ../outputs .

# 2. Git LFS 추적 설정 (대용량 파일)
git lfs track "*.csv" "*.json" "*.tar" "*.pdf" "*.png"

# 3. 불필요한 파일 제외 (.gitignore)
echo "*.safetensors" >> .gitignore
echo "*.pth" >> .gitignore
echo "*.ckpt" >> .gitignore
```

## 4. 업로드 (Push)
이제 Git 명령어로 업로드합니다. 용량이 커서 시간이 걸리지만, 실패해도 다시 `git push` 하면 이어서 됩니다.

```bash
git add .
git commit -m "Upload 19GB dataset"
git push
```

## 5. 완료 후 연결
업로드가 끝나면 `streamlit_app.py`에 URL을 업데이트하세요.

```python
base_url = "https://huggingface.co/datasets/aeoxxian/Datacenter_train/resolve/main/outputs"
```
