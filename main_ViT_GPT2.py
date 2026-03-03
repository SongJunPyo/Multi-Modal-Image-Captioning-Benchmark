import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import shutil
import numpy as np
from transformers import ViTModel, GPT2LMHeadModel, GPT2Tokenizer

# GPU 선택
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_PATH = os.path.join(BASE_DIR, "data", "TRAIN", "images")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "data", "TRAIN", "train.csv")
VAL_IMG_PATH = os.path.join(BASE_DIR, "data", "VAL", "images")
VAL_CSV_PATH = os.path.join(BASE_DIR, "data", "VAL", "val.csv")
SAVE_PATH = os.path.join(BASE_DIR, "results", "generated.csv")
REFERENCE_SAVE_PATH = os.path.join(BASE_DIR, "results", "reference.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results", "model_vit_gpt2.pth")

# 하이퍼파라미터
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # GPT-2는 사전학습 모델이므로 학습률을 매우 낮춰야 함 (중요!)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50

print(f"Using device: {DEVICE}")

# 결과 디렉토리 생성
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 1. GPT-2 Tokenizer 로드
print("Loading GPT-2 Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# GPT-2에는 기본적으로 pad token이 없으므로 eos_token을 pad_token으로 설정
tokenizer.pad_token = tokenizer.eos_token

# 2. 데이터셋 (GPT-2 Tokenizer 적용)
class VLMDataset(Dataset):
    def __init__(self, img_dir, csv_file, tokenizer, transform=None, is_train=True):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.captions = self.df["caption"].tolist()
        self.imgs = self.df["filename"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            caption = self.captions[idx]
            # GPT-2 Tokenizer로 인코딩 
            caption = f"{caption} {self.tokenizer.eos_token}"
            tokens = self.tokenizer(caption, return_tensors='pt', padding=False, truncation=True, max_length=MAX_LEN)
            input_ids = tokens['input_ids'].squeeze(0)
            return image, input_ids
        else:
            return image, img_id

class CollateFn:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        if isinstance(batch[0][1], torch.Tensor):
            targets = [item[1] for item in batch]
            # Batch First=True로 패딩 (GPT-2는 batch_first 선호)
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_token_id)
            return imgs, targets
        else:
            ids = [item[1] for item in batch]
            return imgs, ids

# 3. 모델 정의
class ViTGPT2(nn.Module):
    def __init__(self):
        super(ViTGPT2, self).__init__()
        # Encoder: ViT
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Decoder: GPT-2
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_hidden_size = self.gpt.config.n_embd # 768
        
        # Projection Layer: ViT feature -> GPT-2 embedding space
        self.project = nn.Linear(768, self.gpt_hidden_size)
        
    def forward(self, images, captions):
        # 1. Image Features 뽑기
        # images: (B, 3, 224, 224)
        encoder_out = self.encoder(images).last_hidden_state # (B, 197, 768)
        
        # 2. Projection (이미지 특징을 텍스트 임베딩처럼 변환)
        # image_embeds: (B, 197, 768)
        image_embeds = self.project(encoder_out)
        
        # 3. Caption Embeddings
        # captions: (B, Seq_Len)
        text_embeds = self.gpt.transformer.wte(captions) # (B, Seq_Len, 768)
        
        # 4. Concatenate: [Image, Text] 순서로 붙임
        # inputs_embeds: (B, 197 + Seq_Len, 768)
        inputs_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        
        # 5. GPT-2 Forward
        # labels를 inputs_embeds와 맞춰줘야 함 (Image 부분은 손실 계산 X)
        outputs = self.gpt(inputs_embeds=inputs_embeds)
        
        return outputs.logits # (B, 197 + Seq_Len, Vocab_Size)

    @torch.no_grad()
    def generate_caption(self, images, tokenizer, max_len=30, beam_size=3):
        # 추론용 함수
        B = images.shape[0]
        encoder_out = self.encoder(images).last_hidden_state
        image_embeds = self.project(encoder_out) # (B, 197, 768)
        
        # 생성 시작 토큰 (GPT2는 문맥 없으면 생성이 안되므로 BOS 역할로 EOS 토큰 사용)
        bos_tokens = torch.full((B, 1), tokenizer.eos_token_id, device=images.device)
        bos_embeds = self.gpt.transformer.wte(bos_tokens)
        
        # 입력: [Image, BOS]
        initial_inputs = torch.cat((image_embeds, bos_embeds), dim=1)
        
        # Attention Mask 생성 (Image + BOS)
        attention_mask = torch.ones(initial_inputs.shape[:2], device=images.device)
        
        generated_ids = []
        
        # 간단한 Greedy Decoding 예시 (Beam Search 구현이 복잡하므로 여기선 Greedy로 진행)
        curr_inputs = initial_inputs
        
        for _ in range(max_len):
            outputs = self.gpt(inputs_embeds=curr_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            generated_ids.append(next_token)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # 다음 입력을 위해 임베딩
            next_embed = self.gpt.transformer.wte(next_token)
            curr_inputs = torch.cat((curr_inputs, next_embed), dim=1)
            
        return [tokenizer.decode(g.squeeze().tolist()) for g in generated_ids]


def train_model(model, loader, optimizer, tokenizer):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # 패딩 및 이미지 부분 무시
    
    for idx, (imgs, captions) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        captions = captions.to(DEVICE) # (B, Seq)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(imgs, captions) # (B, 197 + Seq, Vocab)
        
        # Labels 만들기
        # 입력: [Image(197개), Word1, Word2, ..., EOS, PAD]
        # 예측: [   무시    , Word2, Word3, ..., EOS, PAD, ...]
        # 즉, Image 영역과 마지막 토큰은 예측 대상에서 제외하고 shift 시켜야 함
        
        # 1. 전체 레이블 생성 (Image 부분은 -100으로 채움)
        img_len = 197
        batch_size = captions.shape[0]
        seq_len = captions.shape[1]
        
        ignore_prefix = torch.full((batch_size, img_len), -100, device=DEVICE, dtype=torch.long)
        labels = torch.cat((ignore_prefix, captions), dim=1) # (B, 197 + Seq)
        
        # 2. Shift: GPT 모델은 i번째 입력으로 i+1번째를 예측
        # Logits: [..., :-1, :] -> 마지막 예측은 정답이 없으므로 버림
        # Labels: [..., 1:]     -> 첫 번째(Image start)는 예측 안 하므로 버림
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if idx % 50 == 0:
            print(f"Batch {idx}/{len(loader)} Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def generate_captions_gpt2(model, val_loader, tokenizer):
    model.eval()
    results = []
    print(f"Generating captions for {len(val_loader.dataset)} images...")
    
    with torch.no_grad():
        for idx, (imgs, img_ids) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            
            # 배치 단위 추론
            # 구현 편의상 Batch=1로 하거나, 위의 generate_caption 함수를 배치 처리하게 수정해야 함.
            # 여기서는 배치 처리가 가능한 generate_caption 로직을 사용
            
            # Generate 함수 호출 (Greedy 방식)
            for i in range(imgs.size(0)):
                # 단일 이미지 처리
                img_single = imgs[i].unsqueeze(0)
                
                # 수동 생성 루프
                encoder_out = model.encoder(img_single).last_hidden_state
                image_embeds = model.project(encoder_out)
                
                # Start Token
                curr_input_embeds = image_embeds
                generated_tokens = []
                
                for _ in range(40): # Max Len
                    outputs = model.gpt(inputs_embeds=curr_input_embeds)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Greedy
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                        
                    generated_tokens.append(next_token_id.item())
                    
                    next_embed = model.gpt.transformer.wte(next_token_id.unsqueeze(0))
                    curr_input_embeds = torch.cat((curr_input_embeds, next_embed), dim=1)

                caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append({"filename": img_ids[i], "caption": caption})
            
            if idx % 50 == 0:
                print(f"Processed {idx * val_loader.batch_size} images...")

    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"Results saved to {SAVE_PATH}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. 전처리 (ViT 입력용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ViT는 224x224 고정
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # GPT-2 관련 ImageNet Norm or 0.5 mean
    ])
    
    # 2. 데이터셋
    train_dataset = VLMDataset(TRAIN_IMG_PATH, TRAIN_CSV_PATH, tokenizer, transform=transform, is_train=True)
    
    # GPT-2는 Batch First가 편함
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=CollateFn(pad_token_id=tokenizer.eos_token_id)
    )
    
    if os.path.exists(VAL_CSV_PATH):
        val_dataset = VLMDataset(VAL_IMG_PATH, VAL_CSV_PATH, tokenizer, transform=transform, is_train=False)
    else:
        # Dummy Val 생성 로직 (생략 - 기존과 동일)
        pass 

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=CollateFn(pad_token_id=tokenizer.eos_token_id)
    )
    
    # 3. 모델 초기화
    model = ViTGPT2().to(DEVICE)
    
    # 학습 파라미터 설정 (GPT-2 파라미터 + Projection Layer)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 4. 학습
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Start Training ViT + GPT-2...")
        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, optimizer, tokenizer)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    # 5. 생성 및 평가
    generate_captions_gpt2(model, val_loader, tokenizer)
    
    if os.path.exists(VAL_CSV_PATH):
        shutil.copy(VAL_CSV_PATH, REFERENCE_SAVE_PATH)
        print("Running Evaluation...")
        try:
            os.system(f"python {os.path.join(BASE_DIR, 'compute_metrics.py')}")
        except Exception as e:
            print(e)