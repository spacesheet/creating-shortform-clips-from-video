#!/usr/bin/env python3
"""
억양 분석 자동 슬로우모션 (모든 영상 해상도 지원)
세로/가로 영상 자동 감지!
"""

import warnings
warnings.filterwarnings("ignore")

import subprocess
import numpy as np
from pathlib import Path
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import whisper
import json

class ProsodyAnalyzer:
    def __init__(self, input_folder, output_folder=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder
        self.video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        
        print("\n" + "="*70)
        print("🎬 억양 분석 자동 슬로우모션")
        print("="*70)
        print("\n⏳ AI 모델 로딩 중...", end="", flush=True)
        
        self.whisper_model = whisper.load_model("base")
        print(" ✅\n")
    
    def find_videos(self):
        videos = []
        for ext in self.video_extensions:
            videos.extend(self.input_folder.glob(f'*{ext}'))
            videos.extend(self.input_folder.glob(f'*{ext.upper()}'))
        return sorted(set(videos))
    
    def get_video_dimensions(self, video_path):
        """영상 크기 확인"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        width = info['streams'][0]['width']
        height = info['streams'][0]['height']
        return width, height
    
    def get_crop_filter(self, width, height):
        """영상 크기에 맞는 크롭 필터 생성"""
        # 목표: 1080x1920 (9:16)
        target_ratio = 9 / 16  # 가로/세로
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            # 가로가 더 넓음 → 가로를 잘라야 함
            new_width = int(height * target_ratio)
            return f"crop={new_width}:{height},scale=1080:1920"
        else:
            # 세로가 더 김 또는 적절함 → 세로를 잘라야 함 또는 그냥 스케일
            new_height = int(width / target_ratio)
            if new_height <= height:
                return f"crop={width}:{new_height},scale=1080:1920"
            else:
                # 이미 세로가 충분히 김
                return "scale=1080:1920"
    
    def extract_audio(self, video_path):
        """오디오 추출"""
        print("  [1/6] 📀 음성 추출 중...", end="", flush=True)
        
        audio_path = self.output_folder / f"temp_audio_{video_path.stem}.wav"
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            '-y', str(audio_path),
            '-loglevel', 'quiet'
        ]
        
        subprocess.run(cmd)
        print(" ✅")
        return audio_path
    
    def analyze_all(self, audio_path, video_path):
        """모든 분석 한 번에"""
        print("  [2/6] 🎤 억양 분석 중...", end="", flush=True)
        
        all_segments = []
        
        # 1. 피치 분석
        try:
            sound = parselmouth.Sound(str(audio_path))
            pitch = call(sound, "To Pitch", 0.0, 75, 500)
            
            pitch_values = []
            times = []
            
            for t in np.arange(0, sound.duration, 0.01):
                pitch_value = call(pitch, "Get value at time", t, "Hertz", "Linear")
                if pitch_value and not np.isnan(pitch_value):
                    pitch_values.append(pitch_value)
                    times.append(t)
            
            if pitch_values:
                pitch_values = np.array(pitch_values)
                times = np.array(times)
                pitch_change = np.abs(np.diff(pitch_values))
                pitch_change = np.concatenate([[0], pitch_change])
                pitch_change_smooth = gaussian_filter1d(pitch_change, sigma=3)
                
                threshold = np.percentile(pitch_change_smooth, 90)
                peaks, _ = find_peaks(pitch_change_smooth, height=threshold, distance=50)
                
                for peak in peaks:
                    if peak < len(times):
                        all_segments.append({
                            'start': max(0, times[peak] - 0.3),
                            'end': min(sound.duration, times[peak] + 0.7),
                            'score': 3
                        })
        except:
            pass
        
        # 2. 볼륨 분석
        try:
            y, sr = librosa.load(str(audio_path), sr=22050)
            rms = librosa.feature.rms(y=y, frame_length=int(0.1*sr), hop_length=int(0.05*sr))[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=int(0.05*sr))
            rms_smooth = gaussian_filter1d(rms, sigma=3)
            threshold = np.percentile(rms_smooth, 80)
            
            in_emphasis = False
            start_time = 0
            for time, vol in zip(times, rms_smooth):
                if vol > threshold and not in_emphasis:
                    start_time = time
                    in_emphasis = True
                elif vol <= threshold and in_emphasis:
                    if time - start_time >= 0.3:
                        all_segments.append({
                            'start': max(0, start_time - 0.2),
                            'end': time + 0.3,
                            'score': 2
                        })
                    in_emphasis = False
        except:
            pass
        
        # 3. 말하기 속도
        try:
            result = self.whisper_model.transcribe(str(video_path), language="ko", word_timestamps=True, verbose=False)
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment and len(segment.get('words', [])) >= 3:
                        duration = segment['end'] - segment['start']
                        wps = len(segment['words']) / duration
                        if wps > 3.5:
                            all_segments.append({
                                'start': segment['start'],
                                'end': segment['end'],
                                'score': 4
                            })
        except:
            pass
        
        print(" ✅")
        return all_segments
    
    def merge_segments(self, segments):
        """구간 병합"""
        print("  [3/6] 🎯 강조 구간 선정 중...", end="", flush=True)
        
        if not segments:
            print(" ⚠️  (강조 구간 없음)")
            return []
        
        segments.sort(key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            if seg['start'] <= current['end'] + 0.5:
                current['end'] = max(current['end'], seg['end'])
                current['score'] += seg['score']
            else:
                merged.append(current)
                current = seg.copy()
        merged.append(current)
        
        merged.sort(key=lambda x: x['score'], reverse=True)
        top = merged[:6]
        top.sort(key=lambda x: x['start'])
        
        print(f" ✅ ({len(top)}개 구간)")
        return top
    
    def create_video_with_slowmo(self, video_path, segments):
        """영상 생성 (슬로우모션 적용)"""
        output_name = video_path.stem + "_effect.mp4"
        output_path = self.output_folder / output_name
        
        print("  [4/6] 📹 영상 처리 중...")
        
        # 영상 정보
        width, height = self.get_video_dimensions(video_path)
        print(f"     → 원본 크기: {width}x{height}")
        
        crop_filter = self.get_crop_filter(width, height)
        print(f"     → 필터: {crop_filter}")
        
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())
        
        if not segments:
            print("     → 기본 효과만 적용")
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', f'{crop_filter},eq=contrast=1.15:saturation=1.2',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-y', str(output_path),
                '-loglevel', 'error'
            ]
            subprocess.run(cmd)
            return output_path
        
        # 슬로우모션 적용
        for i, seg in enumerate(segments, 1):
            print(f"     → 구간 {i}: {seg['start']:.1f}초 ~ {seg['end']:.1f}초")
        
        segments_all = []
        current = 0
        
        for seg in segments:
            if current < seg['start']:
                segments_all.append({'start': current, 'end': seg['start'], 'speed': 1.0})
            segments_all.append({'start': seg['start'], 'end': seg['end'], 'speed': 0.6})
            current = seg['end']
        
        if current < duration:
            segments_all.append({'start': current, 'end': duration, 'speed': 1.0})
        
        # 각 구간 처리
        temp_dir = self.output_folder / "temp_segments"
        temp_dir.mkdir(exist_ok=True)
        
        segment_files = []
        
        for i, seg in enumerate(segments_all):
            seg_file = temp_dir / f"seg_{i:03d}.mp4"
            
            seg_duration = seg['end'] - seg['start']
            if seg_duration < 0.05:
                continue
            
            speed = seg['speed']
            
            if speed == 1.0:
                cmd = [
                    'ffmpeg', '-ss', str(seg['start']), '-i', str(video_path),
                    '-t', str(seg_duration),
                    '-vf', f'{crop_filter},eq=contrast=1.15:saturation=1.2',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-y', str(seg_file),
                    '-loglevel', 'error'
                ]
            else:
                cmd = [
                    'ffmpeg', '-ss', str(seg['start']), '-i', str(video_path),
                    '-t', str(seg_duration),
                    '-vf', f"setpts={1/speed}*PTS,{crop_filter},eq=contrast=1.15:saturation=1.2",
                    '-af', f"atempo={speed}",
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-y', str(seg_file),
                    '-loglevel', 'error'
                ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            # 세그먼트가 제대로 생성되었는지 확인
            if seg_file.exists() and seg_file.stat().st_size > 1000:
                segment_files.append(seg_file)
            else:
                print(f"     ⚠️  구간 {i} 생성 실패")
        
        if not segment_files:
            print("     ❌ 모든 구간 생성 실패")
            return None
        
        print("  [5/6] 🔗 구간 병합 중...", end="", flush=True)
        
        # concat 파일 생성
        concat_file = temp_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file.absolute()}'\n")
        
        # 병합
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y', str(output_path),
            '-loglevel', 'error'
        ]
        subprocess.run(cmd)
        
        print(" ✅")
        
        # 임시 파일 삭제
        print("  [6/6] 🧹 정리 중...", end="", flush=True)
        for seg_file in segment_files:
            if seg_file.exists():
                seg_file.unlink()
        if concat_file.exists():
            concat_file.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except:
                pass
        
        print(" ✅")
        
        return output_path
    
    def process_single_video(self, video_path):
        """단일 영상 처리"""
        try:
            print(f"\n{'='*70}")
            print(f"📹 {video_path.name}")
            print(f"{'='*70}\n")
            
            # 1. 오디오 추출
            audio_path = self.extract_audio(video_path)
            
            # 2. 억양 분석
            segments = self.analyze_all(audio_path, video_path)
            
            # 3. 구간 병합
            final_segments = self.merge_segments(segments)
            
            # 4. 영상 생성
            output_path = self.create_video_with_slowmo(video_path, final_segments)
            
            # 5. 임시 오디오 삭제
            if audio_path.exists():
                audio_path.unlink()
            
            if output_path and output_path.exists():
                print(f"\n  ✅ 완료!")
                print(f"  📂 저장: {output_path}")
                print(f"{'='*70}\n")
                return True
            else:
                print(f"\n  ❌ 영상 생성 실패")
                print(f"{'='*70}\n")
                return False
            
        except Exception as e:
            print(f"\n  ❌ 오류: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all(self):
        """모든 영상 처리"""
        videos = self.find_videos()
        
        if not videos:
            print("❌ 비디오 파일이 없습니다.")
            return
        
        print(f"📁 총 {len(videos)}개 영상")
        print(f"📂 입력: {self.input_folder}")
        print(f"📂 출력: {self.output_folder}\n")
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        success = 0
        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}]", end=" ")
            if self.process_single_video(video):
                success += 1
        
        print(f"\n{'='*70}")
        print(f"🎉 완료: {success}/{len(videos)}개")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    processor = ProsodyAnalyzer(
        input_folder="./topic_shorts"  # ← 폴더 경로
    )
    
    processor.process_all()
