#!/usr/bin/env python3
"""
쇼츠 영상 자동 자막 생성기
폴더 내의 모든 영상 파일에 자동으로 자막을 생성하고 입혀줍니다.
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def check_dependencies():
    """필요한 프로그램 및 라이브러리 확인"""
    try:
        import whisper
    except ImportError as e:
        print(f"❌ Whisper가 설치되어 있지 않습니다: {e}")
        print("\n다음 명령어로 설치해주세요:")
        print("pip install openai-whisper")
        sys.exit(1)
    
    # ffmpeg 확인
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg가 설치되어 있지 않습니다.")
        print("다음 명령어로 설치해주세요:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    return whisper

def extract_audio_and_transcribe(video_path, model):
    """영상에서 음성을 추출하고 텍스트로 변환"""
    print(f"  🎤 음성 인식 중...")
    
    # Whisper로 직접 영상 파일 처리 (자동으로 오디오 추출)
    result = model.transcribe(
        str(video_path),
        language='ko',  # 한국어로 지정 (자동 감지도 가능)
        verbose=False,
        word_timestamps=False,  # 단어별 타임스탬프 비활성화 (더 긴 세그먼트)
        condition_on_previous_text=True  # 문맥 유지
    )
    
    return result

def merge_short_segments(segments, max_chars=30, min_duration=0.5):
    """최소 병합으로 짧고 빠른 자막 생성"""
    if not segments:
        return []
    
    merged = []
    current = {
        'start': segments[0]['start'],
        'end': segments[0]['end'],
        'text': segments[0]['text'].strip()
    }
    
    for segment in segments[1:]:
        text = segment['text'].strip()
        if not text:
            continue
            
        duration = current['end'] - current['start']
        combined_text = current['text'] + ' ' + text
        time_gap = segment['start'] - current['end']
        
        # 최소 병합 조건 (매우 짧은 자막만 병합):
        # 1. 현재 자막이 0.5초 미만이고
        # 2. 시간 간격이 0.3초 이내이고
        # 3. 합쳐도 30자 이하일 때만
        should_merge = (
            duration < min_duration and 
            time_gap < 0.3 and 
            len(combined_text) <= max_chars
        )
        
        if should_merge:
            current['text'] = combined_text
            current['end'] = segment['end']
        else:
            merged.append(current)
            current = {
                'start': segment['start'],
                'end': segment['end'],
                'text': text
            }
    
    merged.append(current)
    return merged

def fix_overlapping_subtitles(segments):
    """겹치는 자막 완전 제거 - 빠른 전환"""
    if not segments:
        return []
    
    fixed = []
    for i, seg in enumerate(segments):
        current = seg.copy()
        
        # 이전 자막과 겹치는지 확인
        if fixed and current['start'] < fixed[-1]['end']:
            # 이전 자막 끝 + 0.1초 후에 시작 (빠른 전환)
            current['start'] = fixed[-1]['end'] + 0.1
        
        # 다음 자막과 겹치지 않도록 조정
        if i < len(segments) - 1:
            next_start = segments[i + 1]['start']
            if current['end'] > next_start - 0.1:
                # 다음 자막 시작 0.1초 전에 종료
                current['end'] = next_start - 0.1
        
        # 최소 표시 시간 보장 (0.5초 - 빠른 템포)
        min_duration = 0.5
        if current['end'] - current['start'] < min_duration:
            current['end'] = current['start'] + min_duration
        
        # 시작 시간이 종료 시간보다 늦으면 스킵
        if current['start'] >= current['end']:
            continue
            
        fixed.append(current)
    
    return fixed

def create_subtitle_file(result, output_path):
    """자막 파일(.srt) 생성 - 짧고 빠른 자막"""
    # 1단계: 최소 병합 (짧은 자막 유지)
    segments = merge_short_segments(result['segments'])
    
    # 2단계: 겹침 완전 제거
    segments = fix_overlapping_subtitles(segments)
    
    # SRT 파일로 생성
    srt_path = output_path.with_suffix('.srt')
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            if not text:
                continue
            
            # 짧은 자막 유지를 위해 줄바꿈 기준 줄임 (25자 기준)
            if len(text) > 25:
                words = text.split()
                mid = len(words) // 2
                text = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    return srt_path

def format_timestamp(seconds):
    """초를 SRT 타임스탬프 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def add_subtitles_to_video(video_path, subtitle_path, output_path):
    """FFmpeg를 사용해 영상에 자막 입히기 - SRT with strong style override"""
    print(f"  🎬 자막을 영상에 합성 중...")
    
    # 절대 경로로 변환 (Windows 경로 문제 방지)
    subtitle_path_str = str(subtitle_path.absolute()).replace('\\', '\\\\').replace(':', '\\:')
    
    # 큰 글씨 + 중앙보다 살짝 아래 위치
    # MarginV=600 (1920px 기준 하단에서 600px 위)
    style = (
        'FontName=Arial,'
        'FontSize=36,'
        'Bold=1,'
        'PrimaryColour=&H00FFFFFF,'
        'OutlineColour=&H00000000,'
        'BorderStyle=1,'
        'Outline=3,'
        'Shadow=1,'
        'MarginV=600,'
        'Alignment=2'
    )
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f"subtitles={subtitle_path_str}:force_style='{style}'",
        '-c:a', 'copy',
        '-y',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ FFmpeg 오류:")
        print(f"  {e.stderr}")
        return False

def process_video(video_path, output_dir, model):
    """단일 영상 처리"""
    video_name = video_path.stem
    print(f"\n📹 처리 중: {video_path.name}")
    
    try:
        # 1. 음성 인식
        result = extract_audio_and_transcribe(video_path, model)
        
        # 2. 자막 파일 생성 (SRT 형식)
        srt_path_base = output_dir / f"{video_name}.srt"
        srt_path = create_subtitle_file(result, srt_path_base)
        print(f"  ✅ 자막 파일 생성: {srt_path.name}")
        
        # 3. 자막을 영상에 입히기
        output_video_path = output_dir / f"{video_name}_subtitled.mp4"
        success = add_subtitles_to_video(video_path, srt_path, output_video_path)
        
        if success:
            print(f"  ✅ 완성: {output_video_path.name}")
            return True
        else:
            print(f"  ❌ 실패: {video_path.name}")
            return False
            
    except Exception as e:
        print(f"  ❌ 오류 발생: {e}")
        return False

def process_folder(input_folder, output_folder=None, model_size='base'):
    """폴더 내의 모든 영상 처리"""
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {input_folder}")
        sys.exit(1)
    
    # 출력 폴더 설정
    if output_folder is None:
        output_path = input_path / "subtitled"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(exist_ok=True)
    print(f"📁 출력 폴더: {output_path}")
    
    # 지원하는 영상 확장자
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
    
    # 영상 파일 찾기
    video_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        print(f"❌ 영상 파일을 찾을 수 없습니다: {input_folder}")
        sys.exit(1)
    
    print(f"\n🎯 총 {len(video_files)}개의 영상 파일 발견")
    print(f"🤖 Whisper 모델 로딩 중... (모델: {model_size})")
    
    # Whisper 모델 로드
    import whisper
    model = whisper.load_model(model_size)
    
    print("✅ 모델 로딩 완료!\n")
    print("="*60)
    
    # 각 영상 처리
    success_count = 0
    for video_file in video_files:
        if process_video(video_file, output_path, model):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"\n🎉 완료! {success_count}/{len(video_files)}개 영상 처리 성공")
    print(f"📂 결과물 위치: {output_path}")

def main():
    """메인 함수"""
    print("="*60)
    print("🎬 쇼츠 자동 자막 생성기")
    print("="*60)
    
    # 의존성 체크
    check_dependencies()
    
    if len(sys.argv) < 2:
        print("\n사용법:")
        print(f"  python {sys.argv[0]} <영상_폴더_경로> [출력_폴더_경로] [모델_크기]")
        print("\n예시:")
        print(f"  python {sys.argv[0]} ./shorts")
        print(f"  python {sys.argv[0]} ./shorts ./output")
        print(f"  python {sys.argv[0]} ./shorts ./output medium")
        print("\n모델 크기: tiny, base, small, medium, large")
        print("  - tiny/base: 빠르지만 정확도 낮음")
        print("  - small/medium: 균형잡힌 성능 (추천)")
        print("  - large: 느리지만 가장 정확함")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    model_size = sys.argv[3] if len(sys.argv) > 3 else 'base'
    
    # 처리 시작
    process_folder(input_folder, output_folder, model_size)

if __name__ == "__main__":
    main()