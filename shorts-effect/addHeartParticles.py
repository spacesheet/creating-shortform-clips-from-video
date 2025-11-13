import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
import subprocess
import os

# 하트 이미지 생성 함수
def create_heart_image(size=60, color=(255, 105, 180)):
    """PNG 형식의 하트 이미지 생성 (투명 배경)"""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 하트 모양 그리기
    heart_points = []
    for i in range(360):
        angle = i * np.pi / 180
        x = 16 * np.sin(angle) ** 3
        y = -(13 * np.cos(angle) - 5 * np.cos(2*angle) - 2 * np.cos(3*angle) - np.cos(4*angle))
        heart_points.append((size/2 + x*1.5, size/2 + y*1.5))
    
    draw.polygon(heart_points, fill=color + (255,))
    return np.array(img)

# 하트 파티클 클래스
class HeartParticle:
    def __init__(self, video_width, video_height):
        self.start_x = random.randint(0, video_width)
        self.start_y = random.randint(video_height//2, video_height + 100)
        self.size = random.randint(30, 70)
        self.speed_y = random.uniform(-100, -50)  # 위로 올라가는 속도
        self.speed_x = random.uniform(-15, 15)   # 좌우 움직임
        
        # 랜덤 핑크/레드 계열 색상 (BGR for OpenCV)
        colors = [
            (180, 105, 255),  # Hot Pink
            (193, 182, 255),  # Light Pink
            (147, 20, 255),   # Deep Pink
            (203, 192, 255),  # Pink
            (60, 20, 220),    # Crimson
        ]
        self.color = random.choice(colors)
        self.heart_img = create_heart_image(self.size, self.color[::-1])  # RGB to BGR
    
    def get_position(self, t):
        """시간 t에서의 위치 계산 (t는 0~2초)"""
        x = int(self.start_x + self.speed_x * t)
        y = int(self.start_y + self.speed_y * t)
        opacity = max(0, min(1, 1 - t/2))  # 2초 동안 서서히 페이드아웃
        return x, y, opacity

def overlay_heart(frame, heart_img, x, y, opacity):
    """프레임에 하트 이미지 오버레이"""
    h_img, w_img = heart_img.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    
    # 화면 밖이면 스킵
    if x < -w_img or x > w_frame or y < -h_img or y > h_frame:
        return frame
    
    # 경계 처리
    x1_frame = max(0, x)
    y1_frame = max(0, y)
    x2_frame = min(w_frame, x + w_img)
    y2_frame = min(h_frame, y + h_img)
    
    x1_img = max(0, -x)
    y1_img = max(0, -y)
    x2_img = x1_img + (x2_frame - x1_frame)
    y2_img = y1_img + (y2_frame - y1_frame)
    
    if x2_frame <= x1_frame or y2_frame <= y1_frame:
        return frame
    
    # 알파 채널 적용
    alpha = heart_img[y1_img:y2_img, x1_img:x2_img, 3] / 255.0 * opacity
    alpha = alpha[:, :, np.newaxis]
    
    # 블렌딩
    roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]
    heart_rgb = cv2.cvtColor(heart_img[y1_img:y2_img, x1_img:x2_img], cv2.COLOR_RGBA2BGR)
    
    blended = (alpha * heart_rgb + (1 - alpha) * roi).astype(np.uint8)
    frame[y1_frame:y2_frame, x1_frame:x2_frame] = blended
    
    return frame

# ========== 여기서부터 수정하세요! ==========
input_path = "./upload/우리는 플라토닉이지.mp4"  # 입력 비디오 파일 경로
output_path = "./upload/우리는_플라토닉이지_하트효과.mp4"  # 출력 파일 경로

# 효과 설정
effect_start_time = 9.5  # 효과 시작 시간 (초)
effect_end_time = 11.5    # 효과 끝 시간 (초)
num_particles = 25        # 하트 개수
# =========================================

# 파일 존재 확인
if not os.path.exists(input_path):
    print(f"❌ 오류: 파일을 찾을 수 없습니다: {input_path}")
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"올바른 파일 경로를 입력하세요!")
    exit(1)

# 임시 파일 경로
audio_file = output_path.replace('.mp4', '_audio.aac')
video_no_audio = output_path.replace('.mp4', '_video_only.mp4')

print("=" * 50)
print("STEP 1: 원본에서 오디오 추출")
print("=" * 50)
try:
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-vn',  # 비디오 제거
        '-acodec', 'copy',  # 오디오 그대로 복사
        '-y', audio_file
    ], check=True, capture_output=True)
    print(f"✓ 오디오 추출 완료: {audio_file}")
except Exception as e:
    print(f"⚠️ 오디오 추출 실패: {e}")
    print("오디오 없이 진행합니다...")
    audio_file = None

print("\n" + "=" * 50)
print("STEP 2: 비디오 로딩 및 정보 확인")
print("=" * 50)
cap = cv2.VideoCapture(input_path)

# 비디오가 제대로 열렸는지 확인
if not cap.isOpened():
    print(f"❌ 오류: 비디오 파일을 열 수 없습니다: {input_path}")
    exit(1)

# 비디오 정보
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✓ 비디오 로드 성공!")
print(f"  크기: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  총 프레임: {total_frames}")
print(f"  길이: {total_frames/fps:.1f}초")

# 하트 파티클 생성
particles = [HeartParticle(width, height) for _ in range(num_particles)]
print(f"\n{num_particles}개의 하트 파티클 생성 완료! 💕")

print("\n" + "=" * 50)
print("STEP 3: 하트 효과 적용")
print("=" * 50)

# 비디오 작성기 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_no_audio, fourcc, fps, (width, height))

# 효과 적용 구간
effect_start_frame = int(effect_start_time * fps)
effect_end_frame = int(effect_end_time * fps)

print(f"효과 적용 구간: {effect_start_time}초 ~ {effect_end_time}초")
print(f"(프레임 {effect_start_frame} ~ {effect_end_frame})")
print("비디오 처리 중...\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 효과 적용 구간
    if effect_start_frame <= frame_count < effect_end_frame:
        # 현재 시간 (효과 시작부터 경과 시간)
        t = (frame_count - effect_start_frame) / fps
        
        # 모든 파티클 그리기
        for particle in particles:
            x, y, opacity = particle.get_position(t)
            frame = overlay_heart(frame, particle.heart_img, x, y, opacity)
    
    out.write(frame)
    frame_count += 1
    
    if frame_count % 300 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"  진행률: {progress:.1f}% ({frame_count}/{total_frames} 프레임)")

print("✓ 비디오 처리 완료!")
cap.release()
out.release()

print("\n" + "=" * 50)
print("STEP 4: 비디오 + 오디오 합치기")
print("=" * 50)

if audio_file and os.path.exists(audio_file):
    try:
        # 오디오 파일과 비디오 파일을 합치기
        subprocess.run([
            'ffmpeg',
            '-i', video_no_audio,  # 효과 적용된 비디오 (오디오 없음)
            '-i', audio_file,       # 추출한 오디오
            '-c:v', 'libx264',      # 비디오 h264 인코딩
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',          # 오디오 aac 인코딩
            '-b:a', '192k',         # 오디오 비트레이트
            '-shortest',            # 짧은 스트림에 맞춤
            '-y', output_path
        ], check=True, capture_output=True)
        
        print("✓ 오디오 병합 완료!")
        
        # 임시 파일 삭제
        if os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(video_no_audio):
            os.remove(video_no_audio)
            
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 병합 실패: {e.stderr}")
        print(f"비디오 파일: {video_no_audio}")
        print(f"오디오 파일: {audio_file}")
else:
    print("⚠️ 오디오 파일이 없습니다. 비디오만 저장됩니다.")
    if os.path.exists(video_no_audio):
        os.rename(video_no_audio, output_path)

print("\n" + "=" * 50)
print("✨ 완료! ✨")
print("=" * 50)
if os.path.exists(output_path):
    print(f"📁 파일 위치: {output_path}")
    print(f"📦 파일 크기: {os.path.getsize(output_path) / (1024*1024):.1f}MB")
else:
    print(f"❌ 출력 파일 생성 실패")