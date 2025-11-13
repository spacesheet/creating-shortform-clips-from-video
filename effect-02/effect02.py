# -*- coding: utf-8 -*-

import os
import json
import subprocess
from datetime import timedelta
from faster_whisper import WhisperModel
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import re

class TopicBasedShortCreator:
    def __init__(self, video_path, output_dir="shorts", similarity_threshold=0.7):
        self.video_path = video_path
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        self.transcription_file = "transcription.json"
        os.makedirs(output_dir, exist_ok=True)
        
        # 의미 임베딩 모델 로드
        print("🤖 문장 임베딩 모델 로드 중...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 비디오 정보 가져오기
        self.video_info = self._get_video_info()
        
    def _get_video_info(self):
        """ffprobe로 비디오 정보 가져오기"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            self.video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            return {
                'duration': float(info['format']['duration']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height'])
            }
        except Exception as e:
            print(f"⚠️  비디오 정보를 가져올 수 없습니다: {e}")
            return {'duration': 0, 'width': 1920, 'height': 1080}
    
    def transcribe_video(self, force_new=False):
        """영상 음성 인식 (faster-whisper 사용)"""
        if os.path.exists(self.transcription_file) and not force_new:
            print("📄 기존 자막 파일 로드 중...")
            with open(self.transcription_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print("🎤 음성 인식 시작... (faster-whisper 사용)")
        print("   ⚡ 기존 Whisper보다 4-5배 빠릅니다!")
        
        # faster-whisper 모델 로드
        # compute_type: "int8" (CPU 최적화), "float16" (GPU), "float32" (정확도 우선)
        model = WhisperModel(
            "base", 
            device="cpu", 
            compute_type="int8"
        )
        
        # 음성 인식 수행
        segments_generator, info = model.transcribe(
            self.video_path,
            language="ko",
            word_timestamps=True,
            vad_filter=True,  # 음성 구간 자동 감지
            beam_size=5
        )
        
        # generator를 리스트로 변환하고 Whisper 형식에 맞게 변환
        segments = []
        for segment in segments_generator:
            segments.append({
                'id': segment.id,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'words': [
                    {
                        'start': word.start,
                        'end': word.end,
                        'word': word.word,
                        'probability': word.probability
                    }
                    for word in (segment.words or [])
                ] if segment.words else []
            })
        
        result = {
            'segments': segments,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration
        }
        
        # 결과 저장
        with open(self.transcription_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 음성 인식 완료! (총 {len(segments)}개 구간)")
        print(f"   언어: {info.language} (확률: {info.language_probability:.2%})")
        print(f"   영상 길이: {info.duration:.1f}초")
        
        return result
    
    def find_topic_boundaries(self, segments, window_size=3):
        """의미적 유사도를 기반으로 화제 전환점 찾기"""
        print(f"\n🔍 화제 전환점 분석 중...")
        
        if len(segments) < 2:
            return [0, len(segments)]
        
        # 각 세그먼트의 텍스트 추출
        texts = [seg['text'].strip() for seg in segments if seg['text'].strip()]
        
        if len(texts) < 2:
            return [0, len(segments)]
        
        # 문장 임베딩 생성
        print("   📊 문장 임베딩 생성 중...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # 연속된 세그먼트 간 유사도 계산
        similarities = []
        for i in range(len(embeddings) - 1):
            start_idx = max(0, i - window_size + 1)
            end_idx = min(len(embeddings), i + window_size + 1)
            
            window1 = np.mean(embeddings[start_idx:i+1], axis=0)
            window2 = np.mean(embeddings[i+1:end_idx], axis=0)
            
            similarity = np.dot(window1, window2) / (
                np.linalg.norm(window1) * np.linalg.norm(window2)
            )
            similarities.append(similarity)
        
        # 유사도가 낮은 지점 = 화제 전환점
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
                print(f"   ✅ 화제 전환 감지: {i+1}번째 구간 (유사도: {sim:.2f})")
        
        boundaries.append(len(segments))
        
        print(f"\n📌 총 {len(boundaries)-1}개의 화제 구간 발견")
        return boundaries
    
    def extract_keywords(self, text, top_n=3):
        """핵심 키워드 추출"""
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        stop_words = ['은', '는', '이', '가', '을', '를', '에', '의', '와', '과', 
                      '도', '으로', '로', '에서', '하다', '있다', '되다', '이다',
                      '그', '저', '것', '수', '등', '및', '좀', '막', '진짜', '되게']
        
        words = text.split()
        words = [w for w in words if len(w) > 1 and w not in stop_words]
        
        if not words:
            return []
        
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(top_n)]
    
    def find_topic_segments(self, transcription, min_duration=15, max_duration=60):
        """
        화제 단위로 모든 구간 분할 (키워드 필터링 없음)
        """
        segments = transcription['segments']
        
        # 1. 화제 경계 찾기
        boundaries = self.find_topic_boundaries(segments)
        
        # 2. 각 화제를 숏폼 구간으로 변환
        topic_clips = []
        
        print(f"\n✂️  화제별 구간 생성 중 (키워드 필터링 없음 - 모든 화제 포함)...")
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            if start_idx >= len(segments) or end_idx > len(segments):
                continue
            
            # 화제 내 모든 세그먼트 합치기
            topic_segments = segments[start_idx:end_idx]
            
            if not topic_segments:
                continue
            
            # 전체 텍스트
            full_text = ' '.join([s['text'] for s in topic_segments])
            
            start_time = topic_segments[0]['start']
            end_time = topic_segments[-1]['end']
            duration = end_time - start_time
            
            # 길이 조건 확인
            if duration < min_duration:
                print(f"   ⏭️  화제 {i+1}: 너무 짧음 ({duration:.1f}초 < {min_duration}초)")
                continue
            
            # 최대 길이 초과 시 분할
            if duration > max_duration:
                # 긴 화제를 여러 개의 숏폼으로 분할
                sub_clips = self._split_long_topic(
                    topic_segments, 
                    start_time, 
                    end_time, 
                    max_duration
                )
                topic_clips.extend(sub_clips)
            else:
                # 자동 키워드 추출
                keywords = self.extract_keywords(full_text)
                
                topic_clips.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'text': full_text,
                    'keywords': keywords,
                    'topic_id': i + 1
                })
                
                print(f"   ✅ 화제 {i+1}: {duration:.1f}초 | 키워드: {', '.join(keywords)}")
        
        print(f"\n✅ 총 {len(topic_clips)}개의 숏폼 구간 생성 완료")
        return topic_clips
    
    def _split_long_topic(self, segments, start_time, end_time, max_duration):
        """긴 화제를 여러 개의 숏폼으로 분할"""
        clips = []
        current_start = start_time
        current_segments = []
        current_duration = 0
        
        for seg in segments:
            seg_duration = seg['end'] - seg['start']
            
            if current_duration + seg_duration > max_duration and current_segments:
                # 현재 구간 저장
                text = ' '.join([s['text'] for s in current_segments])
                keywords = self.extract_keywords(text)
                
                clips.append({
                    'start': current_start,
                    'end': current_segments[-1]['end'],
                    'duration': current_segments[-1]['end'] - current_start,
                    'text': text,
                    'keywords': keywords,
                    'topic_id': len(clips) + 1
                })
                
                # 새 구간 시작
                current_start = seg['start']
                current_segments = [seg]
                current_duration = seg_duration
            else:
                current_segments.append(seg)
                current_duration += seg_duration
        
        # 마지막 구간 저장
        if current_segments:
            text = ' '.join([s['text'] for s in current_segments])
            keywords = self.extract_keywords(text)
            
            clips.append({
                'start': current_start,
                'end': current_segments[-1]['end'],
                'duration': current_segments[-1]['end'] - current_start,
                'text': text,
                'keywords': keywords,
                'topic_id': len(clips) + 1
            })
        
        return clips
    
    def _format_time(self, seconds):
        """초를 HH:MM:SS 형식으로 변환"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _generate_filename(self, keywords, index, max_length=50):
        """키워드를 기반으로 파일명 생성"""
        if not keywords:
            return f"short_{index:03d}"
        
        # 키워드를 언더스코어로 연결
        keyword_part = "_".join(keywords[:3])  # 최대 3개 키워드 사용
        
        # 파일명에 사용할 수 없는 문자 제거
        keyword_part = re.sub(r'[^\w가-힣]', '_', keyword_part)
        
        # 길이 제한
        if len(keyword_part) > max_length:
            keyword_part = keyword_part[:max_length]
        
        # 번호와 키워드 조합
        return f"short_{index:03d}_{keyword_part}"
    
    def create_shorts(self, clips, vertical=True):
        """숏폼 영상 생성"""
        if not clips:
            return []
        
        print(f"\n🎬 숏폼 영상 생성 중...")
        shorts = []
        
        for i, clip in enumerate(clips, 1):
            # 키워드 기반 파일명 생성
            keywords = clip.get('keywords', [])
            filename_base = self._generate_filename(keywords, i)
            filename = f"{filename_base}.mp4"
            output_path = os.path.join(self.output_dir, filename)
            
            # 세로 영상 여부에 따른 필터 설정
            if vertical:
                # 9:16 세로 비율로 크롭
                crop_filter = f"scale={self.video_info['height']*9//16}:{self.video_info['height']},crop={self.video_info['height']*9//16}:{self.video_info['height']}"
            else:
                crop_filter = "scale=1080:1920"
            
            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(clip['start']),
                '-i', self.video_path,
                '-t', str(clip['duration']),
                '-vf', crop_filter,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                output_path
            ]
            
            try:
                subprocess.run(
                    cmd, 
                    capture_output=True, 
                    check=True,
                    encoding='utf-8'
                )
                
                shorts.append({
                    'filename': filename,
                    'path': output_path,
                    'start': clip['start'],
                    'end': clip['end'],
                    'duration': clip['duration'],
                    'text': clip['text'],
                    'keywords': clip.get('keywords', []),
                    'topic_id': clip.get('topic_id', i)
                })
                
                print(f"   ✅ [{i}/{len(clips)}] {filename} 생성 완료 ({clip['duration']:.1f}초)")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ [{i}/{len(clips)}] {filename} 생성 실패")
                print(f"      에러: {e.stderr}")
        
        return shorts
    
    def create_fcpxml(self, shorts, filename="project.fcpxml"):
        """Final Cut Pro 프로젝트 파일 생성"""
        if not shorts:
            return None
        
        print(f"\n📦 Final Cut Pro 프로젝트 생성 중...")
        
        fcpxml_path = os.path.join(self.output_dir, filename)
        
        # FCPXML 기본 구조
        fcpxml = ET.Element('fcpxml', version='1.9')
        resources = ET.SubElement(fcpxml, 'resources')
        library = ET.SubElement(fcpxml, 'library')
        event = ET.SubElement(library, 'event', name='화제별 숏폼')
        project = ET.SubElement(event, 'project', name='숏폼 프로젝트')
        sequence = ET.SubElement(project, 'sequence', format='r1', duration=f'{sum(s["duration"] for s in shorts):.2f}s')
        spine = ET.SubElement(sequence, 'spine')
        
        # 각 숏폼을 타임라인에 추가
        for i, short in enumerate(shorts):
            # 리소스 등록
            asset_id = f'r{i+1}'
            ET.SubElement(resources, 'asset', {
                'id': asset_id,
                'name': short['filename'],
                'src': f"file://{os.path.abspath(short['path'])}"
            })
            
            # 클립 추가
            clip = ET.SubElement(spine, 'clip', {
                'name': short['filename'],
                'ref': asset_id,
                'duration': f'{short["duration"]:.2f}s',
                'start': f'{short["start"]:.2f}s'
            })
            
            # 키워드 마커 추가
            keywords_to_mark = short.get('keywords', [])
            if keywords_to_mark:
                keyword_text = ', '.join(keywords_to_mark)
                marker = ET.SubElement(clip, 'marker', {
                    'start': '0s',
                    'duration': '1/30s',
                    'value': keyword_text
                })
        
        xml_str = minidom.parseString(ET.tostring(fcpxml)).toprettyxml(indent="  ")
        
        with open(fcpxml_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        print(f"✅ FCPXML 생성 완료: {fcpxml_path}")
        return fcpxml_path
    
    def save_report(self, shorts, filename="report.txt"):
        """결과 리포트 저장"""
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("화제별 숏폼 생성 리포트 (전체 화제)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"원본 영상: {self.video_path}\n")
            f.write(f"생성된 숏폼: {len(shorts)}개\n")
            f.write(f"유사도 임계값: {self.similarity_threshold}\n")
            f.write(f"필터링 키워드: 없음 (전체 화제 추출)\n\n")
            
            for i, short in enumerate(shorts, 1):
                f.write(f"\n[{i}] {short['filename']}\n")
                if short.get('keywords'):
                    f.write(f"    자동 키워드: {', '.join(short['keywords'])}\n")
                f.write(f"    시간: {self._format_time(short['start'])} ~ {self._format_time(short['end'])}\n")
                f.write(f"    길이: {short['duration']:.1f}초\n")
                f.write(f"    내용: {short['text'][:100]}{'...' if len(short['text']) > 100 else ''}\n")
                f.write("-" * 60 + "\n")
        
        print(f"📄 리포트 저장: {report_path}")


def main():
    """메인 실행 함수"""
    
    # Segmentation fault 방지
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    try:
        torch.set_num_threads(1)
    except ImportError:
        pass
    
    # ==================== 설정 ====================
    VIDEO_PATH = "2025-10-29 먹방_music_removed.mp4"  # 영상 파일 경로
    OUTPUT_DIR = "fast_shorts"      # 출력 폴더
    
    # 화제 감지 설정
    SIMILARITY_THRESHOLD = 0.5  # 낮을수록 더 많은 화제로 분할 (0.5~0.8 권장)
    
    # 숏폼 길이 설정
    MIN_DURATION = 15  # 최소 길이 (초)
    MAX_DURATION = 60  # 최대 길이 (초)
    
    # 세로 영상 여부
    VERTICAL = True  # True: 9:16 세로, False: 16:9 가로
    # =============================================
    
    # ffmpeg 설치 확인
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg가 설치되어 있지 않습니다!")
        print("\n설치 방법:")
        print("  Mac: brew install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 영상 파일을 찾을 수 없습니다: {VIDEO_PATH}")
        return
    
    print("🎬 화제 전환 기반 숏폼 자동 생성기")
    print("=" * 60)
    print("💡 키워드 필터링 없이 모든 화제 구간을 숏폼으로 제작합니다")
    print("=" * 60)
    
    creator = TopicBasedShortCreator(
        VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    # 1. 음성 인식
    transcription = creator.transcribe_video()
    
    # 2. 화제별 구간 찾기 (키워드 필터링 없음)
    topic_clips = creator.find_topic_segments(
        transcription,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION
    )
    
    if not topic_clips:
        print("\n❌ 조건에 맞는 구간을 찾지 못했습니다.")
        print(f"💡 팁: SIMILARITY_THRESHOLD를 조정하거나 MIN_DURATION을 낮춰보세요")
        return
    
    # 3. 숏폼 생성
    shorts = creator.create_shorts(topic_clips, vertical=VERTICAL)
    
    if not shorts:
        print("\n❌ 숏폼을 생성하지 못했습니다.")
        return
    
    # 4. FCP 프로젝트 생성
    fcpxml_path = creator.create_fcpxml(shorts)
    
    # 5. 리포트 저장
    creator.save_report(shorts)
    
    # 완료!
    print("\n" + "=" * 60)
    print("🎉 완료!")
    print("=" * 60)
    print(f"📁 저장 위치: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📊 생성된 숏폼: {len(shorts)}개")
    print(f"\n💡 설정:")
    print(f"   - 화제 전환 임계값: {SIMILARITY_THRESHOLD}")
    print(f"   - 키워드 필터링: OFF (모든 화제 포함)")
    print(f"   - 최소 길이: {MIN_DURATION}초")
    print(f"   - 최대 길이: {MAX_DURATION}초")
    print(f"\n생성된 파일:")
    for i, short in enumerate(shorts[:10], 1):
        keywords = short.get('keywords', [])
        keyword_text = f" | 키워드: {', '.join(keywords)}" if keywords else ""
        print(f"  {short['filename']} ({short['duration']:.1f}초){keyword_text}")
    if len(shorts) > 10:
        print(f"  ... 외 {len(shorts)-10}개")
    print(f"\n🎬 {fcpxml_path}을 Final Cut Pro에서 열어보세요!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        from faster_whisper import WhisperModel
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ 필요한 라이브러리를 설치해주세요:")
        print("\npip install faster-whisper sentence-transformers")
        print("\n💡 faster-whisper는 기존 Whisper보다 4-5배 빠릅니다!")
        exit(1)
    main()