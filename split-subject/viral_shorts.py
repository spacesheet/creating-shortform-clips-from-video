# -*- coding: utf-8 -*-

import os
import json
import subprocess
from datetime import timedelta
import whisper
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

class TopicBasedShortCreator:
    def __init__(self, video_path, keywords=None, output_dir="shorts", similarity_threshold=0.7):
        self.video_path = video_path
        self.keywords = [k.lower() for k in keywords] if keywords else []
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
        """영상 음성 인식"""
        if os.path.exists(self.transcription_file) and not force_new:
            print("📄 기존 자막 파일 로드 중...")
            with open(self.transcription_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print("🎤 음성 인식 시작... (시간이 좀 걸릴 수 있습니다)")
        model = whisper.load_model("base", device="cpu")
        
        result = model.transcribe(
            self.video_path,
            language="ko",
            word_timestamps=True,
            verbose=False,
            fp16=False
        )
        
        with open(self.transcription_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 음성 인식 완료! (총 {len(result['segments'])}개 구간)")
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
    
    def contains_keywords(self, text, context_window=0):
        """텍스트에 지정된 키워드가 포함되어 있는지 확인"""
        if not self.keywords:
            return True, []
        
        text_lower = text.lower()
        matched = [kw for kw in self.keywords if kw in text_lower]
        return len(matched) > 0, matched
    
    def find_topic_segments(self, transcription, min_duration=15, max_duration=60, 
                           use_keyword_filter=True, context_segments=2):
        """
        화제 단위로 구간 분할
        
        Args:
            use_keyword_filter: 키워드 필터링 사용 여부
            context_segments: 키워드 발견 시 앞뒤로 포함할 구간 수
        """
        segments = transcription['segments']
        
        # 1. 화제 경계 찾기
        boundaries = self.find_topic_boundaries(segments)
        
        # 2. 각 화제를 숏폼 구간으로 변환
        topic_clips = []
        
        print(f"\n✂️  화제별 구간 생성 중...")
        if use_keyword_filter and self.keywords:
            print(f"🔑 키워드 필터링: {', '.join(self.keywords[:10])}" + 
                  (f" 외 {len(self.keywords)-10}개" if len(self.keywords) > 10 else ""))
        
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
            
            # 키워드 필터링
            if use_keyword_filter and self.keywords:
                has_keyword, matched_keywords = self.contains_keywords(full_text)
                
                if not has_keyword:
                    continue  # 키워드가 없으면 스킵
                
                # 컨텍스트 확장 (키워드 주변 구간도 포함)
                extended_start_idx = max(0, start_idx - context_segments)
                extended_end_idx = min(len(segments), end_idx + context_segments)
                topic_segments = segments[extended_start_idx:extended_end_idx]
                full_text = ' '.join([s['text'] for s in topic_segments])
            else:
                matched_keywords = []
            
            start_time = topic_segments[0]['start']
            end_time = topic_segments[-1]['end']
            duration = end_time - start_time
            
            # 핵심 키워드 추출 (자동 + 매칭된 키워드)
            auto_keywords = self.extract_keywords(full_text, top_n=3)
            all_keywords = list(set(matched_keywords + auto_keywords))
            
            # 너무 짧거나 긴 구간 조정
            if duration < min_duration:
                extend = (min_duration - duration) / 2
                start_time = max(0, start_time - extend)
                end_time = min(self.video_info['duration'], end_time + extend)
                duration = end_time - start_time
                
                topic_clips.append({
                    'topic_id': i + 1,
                    'keywords': all_keywords,
                    'matched_keywords': matched_keywords,
                    'text': full_text[:200],
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
                
            elif duration > max_duration:
                split_clips = self._split_long_topic(
                    topic_segments, 
                    max_duration, 
                    topic_id=i + 1,
                    keywords=all_keywords,
                    matched_keywords=matched_keywords
                )
                topic_clips.extend(split_clips)
                
            else:
                topic_clips.append({
                    'topic_id': i + 1,
                    'keywords': all_keywords,
                    'matched_keywords': matched_keywords,
                    'text': full_text[:200],
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
            
            # 진행상황 출력
            if len(topic_clips) <= 50 or len(topic_clips) % 10 == 0:
                print(f"   화제 {i+1}: {start_time:.1f}s ~ {end_time:.1f}s ({duration:.1f}초)")
                if matched_keywords:
                    print(f"      ✅ 매칭: {', '.join(matched_keywords[:3])}")
                elif all_keywords:
                    print(f"      키워드: {', '.join(all_keywords[:3])}")
        
        print(f"\n🎯 총 {len(topic_clips)}개의 숏폼 구간 생성")
        return topic_clips
    
    def _split_long_topic(self, segments, max_duration, topic_id=0, keywords=None, matched_keywords=None):
        """긴 화제를 여러 개의 숏폼으로 분할"""
        clips = []
        current_segments = []
        current_duration = 0
        part_num = 1
        
        if keywords is None:
            keywords = []
        if matched_keywords is None:
            matched_keywords = []
        
        for seg in segments:
            seg_duration = seg['end'] - seg['start']
            
            if current_duration + seg_duration > max_duration and current_segments:
                start_time = current_segments[0]['start']
                end_time = current_segments[-1]['end']
                full_text = ' '.join([s['text'] for s in current_segments])
                clip_keywords = self.extract_keywords(full_text, top_n=2) if not keywords else keywords
                
                clips.append({
                    'topic_id': f"{topic_id}-{part_num}",
                    'keywords': clip_keywords,
                    'matched_keywords': matched_keywords,
                    'text': full_text[:200],
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
                
                current_segments = [seg]
                current_duration = seg_duration
                part_num += 1
            else:
                current_segments.append(seg)
                current_duration += seg_duration
        
        # 마지막 클립
        if current_segments:
            start_time = current_segments[0]['start']
            end_time = current_segments[-1]['end']
            full_text = ' '.join([s['text'] for s in current_segments])
            clip_keywords = self.extract_keywords(full_text, top_n=2) if not keywords else keywords
            
            clips.append({
                'topic_id': f"{topic_id}-{part_num}",
                'keywords': clip_keywords,
                'matched_keywords': matched_keywords,
                'text': full_text[:200],
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
        
        return clips
    
    def create_shorts(self, topic_clips, vertical=True):
        """숏폼 영상 생성"""
        print(f"\n✂️  숏폼 영상 생성 중...")
        
        created_shorts = []
        
        for i, clip_info in enumerate(topic_clips, 1):
            start = clip_info['start']
            end = clip_info['end']
            duration = clip_info['duration']
            
            # 파일명 생성 (매칭된 키워드 우선)
            if clip_info.get('matched_keywords'):
                keyword_name = '_'.join(clip_info['matched_keywords'][:2])
            elif clip_info['keywords']:
                keyword_name = '_'.join(clip_info['keywords'][:2])
            else:
                keyword_name = 'topic'
            
            keyword_name = re.sub(r'[^\w가-힣]', '_', keyword_name)
            topic_id = clip_info.get('topic_id', i)
            output_filename = f"short_{i:03d}_{keyword_name}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            print(f"\n   [{i}/{len(topic_clips)}] {output_filename}")
            print(f"      시간: {self._format_time(start)} ~ {self._format_time(end)} ({duration:.1f}초)")
            if clip_info.get('matched_keywords'):
                print(f"      ✅ 매칭: {', '.join(clip_info['matched_keywords'][:3])}")
            elif clip_info['keywords']:
                print(f"      키워드: {', '.join(clip_info['keywords'][:3])}")
            
            try:
                if vertical:
                    success = self._create_vertical_clip(
                        self.video_path,
                        output_path,
                        start,
                        duration
                    )
                else:
                    success = self._create_clip(
                        self.video_path,
                        output_path,
                        start,
                        duration
                    )
                
                if success:
                    created_shorts.append({
                        'filename': output_filename,
                        'path': os.path.abspath(output_path),
                        'keywords': clip_info['keywords'],
                        'matched_keywords': clip_info.get('matched_keywords', []),
                        'text': clip_info['text'],
                        'start': start,
                        'end': end,
                        'duration': duration
                    })
                    print(f"      ✅ 저장 완료!")
                else:
                    print(f"      ❌ 생성 실패")
                    
            except Exception as e:
                print(f"      ❌ 오류: {e}")
                continue
        
        return created_shorts
    
    def _create_clip(self, input_path, output_path, start, duration):
        """ffmpeg로 영상 자르기 (원본 비율)"""
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _create_vertical_clip(self, input_path, output_path, start, duration):
        """ffmpeg로 9:16 세로 영상 생성"""
        target_width = 1080
        target_height = 1920
        
        current_width = self.video_info['width']
        current_height = self.video_info['height']
        
        scale_for_width = target_height / current_height
        scale_for_height = target_width / current_width
        
        if current_width / current_height > target_width / target_height:
            scale = scale_for_width
            crop_width = int(target_width / scale)
            crop_height = current_height
            crop_x = int((current_width - crop_width) / 2)
            crop_y = 0
        else:
            scale = scale_for_height
            crop_width = current_width
            crop_height = int(target_height / scale)
            crop_x = 0
            crop_y = int((current_height - crop_height) / 2)
        
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(duration),
            '-vf', f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={target_width}:{target_height}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"      ffmpeg 오류: {e}")
            return False
    
    def _format_time(self, seconds):
        """초를 MM:SS 형식으로"""
        return str(timedelta(seconds=int(seconds)))[2:]
    
    def create_fcpxml(self, shorts, output_file="topic_shorts.fcpxml"):
        """FCP XML 프로젝트 파일 생성"""
        print(f"\n📦 Final Cut Pro 프로젝트 생성 중...")
        
        fcpxml_path = os.path.join(self.output_dir, output_file)
        
        fcpxml = ET.Element('fcpxml', version="1.11")
        resources = ET.SubElement(fcpxml, 'resources')
        
        format_elem = ET.SubElement(resources, 'format', {
            'id': 'r0',
            'name': 'FFVideoFormat1080p9x16',
            'frameDuration': '1001/30000s',
            'width': '1080',
            'height': '1920'
        })
        
        for i, short in enumerate(shorts, 1):
            asset = ET.SubElement(resources, 'asset', {
                'id': f'r{i}',
                'name': short['filename'],
                'uid': f'asset-{i}',
                'src': f"file://{short['path']}",
                'start': '0s',
                'duration': f"{short['duration']:.3f}s",
                'hasVideo': '1',
                'hasAudio': '1',
                'format': 'r0'
            })
        
        library = ET.SubElement(fcpxml, 'library')
        event = ET.SubElement(library, 'event', name='화제별 숏폼')
        
        project = ET.SubElement(event, 'project', name='숏폼 모음')
        sequence = ET.SubElement(project, 'sequence', {
            'format': 'r0',
            'tcStart': '0s',
            'tcFormat': 'NDF',
            'audioLayout': 'stereo',
            'audioRate': '48k'
        })
        
        spine = ET.SubElement(sequence, 'spine')
        
        for i, short in enumerate(shorts, 1):
            clip = ET.SubElement(spine, 'asset-clip', {
                'ref': f'r{i}',
                'offset': '0s',
                'name': short['filename'],
                'duration': f"{short['duration']:.3f}s",
                'format': 'r0',
                'tcFormat': 'NDF'
            })
            
            keywords_to_mark = short.get('matched_keywords', []) or short.get('keywords', [])
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
            f.write("화제별 숏폼 생성 리포트\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"원본 영상: {self.video_path}\n")
            f.write(f"생성된 숏폼: {len(shorts)}개\n")
            f.write(f"유사도 임계값: {self.similarity_threshold}\n")
            if self.keywords:
                f.write(f"필터링 키워드: {len(self.keywords)}개\n\n")
            else:
                f.write(f"필터링 키워드: 없음 (전체 화제)\n\n")
            
            for i, short in enumerate(shorts, 1):
                f.write(f"\n[{i}] {short['filename']}\n")
                if short.get('matched_keywords'):
                    f.write(f"    ✅ 매칭 키워드: {', '.join(short['matched_keywords'])}\n")
                if short.get('keywords'):
                    f.write(f"    자동 키워드: {', '.join(short['keywords'])}\n")
                f.write(f"    시간: {self._format_time(short['start'])} ~ {self._format_time(short['end'])}\n")
                f.write(f"    길이: {short['duration']:.1f}초\n")
                f.write(f"    내용: {short['text']}\n")
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
    VIDEO_PATH = "나의 동영상.mov"
    OUTPUT_DIR = "topic_shorts"
    
    # 키워드 필터링 (None이면 전체 화제 추출)
    KEYWORDS = [
        # None으로 설정하면 모든 화제 추출
    ]
    
    USE_KEYWORD_FILTER = True  # False로 설정하면 키워드 무시하고 전체 화제 추출
    CONTEXT_SEGMENTS = 2  # 키워드 발견 시 앞뒤로 포함할 구간 수
    
    VERTICAL = True
    MIN_DURATION = 15
    MAX_DURATION = 60
    SIMILARITY_THRESHOLD = 0.65
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
    
    print("🎬 화제 기반 숏폼 자동 생성기 (키워드 필터링)")
    print("=" * 60)
    
    creator = TopicBasedShortCreator(
        VIDEO_PATH, 
        keywords=KEYWORDS if USE_KEYWORD_FILTER else None,
        output_dir=OUTPUT_DIR,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    # 1. 음성 인식
    transcription = creator.transcribe_video()
    
    # 2. 화제별 구간 찾기
    topic_clips = creator.find_topic_segments(
        transcription,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION,
        use_keyword_filter=USE_KEYWORD_FILTER,
        context_segments=CONTEXT_SEGMENTS
    )
    
    if not topic_clips:
        print("\n❌ 조건에 맞는 구간을 찾지 못했습니다.")
        if USE_KEYWORD_FILTER and KEYWORDS:
            print(f"💡 팁: 키워드를 더 추가하거나 USE_KEYWORD_FILTER=False로 설정해보세요")
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
    print(f"   - 키워드 필터링: {'ON' if USE_KEYWORD_FILTER and KEYWORDS else 'OFF'}")
    if USE_KEYWORD_FILTER and KEYWORDS:
        print(f"   - 필터링 키워드: {len(KEYWORDS)}개")
        print(f"   - 컨텍스트 확장: 앞뒤 {CONTEXT_SEGMENTS}구간")
    print(f"\n생성된 파일:")
    for i, short in enumerate(shorts[:10], 1):
        matched = short.get('matched_keywords', [])
        marker = f" [✅ {', '.join(matched[:2])}]" if matched else ""
        print(f"  {short['filename']} ({short['duration']:.1f}초){marker}")
    if len(shorts) > 10:
        print(f"  ... 외 {len(shorts)-10}개")
    print(f"\n🎬 {fcpxml_path}을 Final Cut Pro에서 열어보세요!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        import whisper
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("❌ 필요한 라이브러리를 설치해주세요:")
        print("\npip install openai-whisper sentence-transformers scikit-learn")
        exit(1)
    main()
